import os
import io
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import YoutubeLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Markdown, display


# Function to get the video title
def get_video_title(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        title_tag = soup.find("title")
        if not title_tag:
            raise ValueError("Could not retrieve the video title.")
        return title_tag.get_text().strip()
    except Exception as e:
        raise RuntimeError(f"Error fetching video title: {e}")


# Function to get the transcript and video title
def get_video_info(url_video, language="en", fallback_language="en-GB", translation=None):
    try:
        # Attempt to load transcript in the primary language
        video_loader = YoutubeLoader.from_youtube_url(
            url_video,
            language=language,
            translation=translation,
        )
        infos = video_loader.load()

        if not infos:  # Retry with fallback language
            video_loader = YoutubeLoader.from_youtube_url(
                url_video,
                language=fallback_language,
                translation=translation,
            )
            infos = video_loader.load()

        if not infos:
            raise ValueError(f"No transcript available for the video: {url_video}")

        transcript = infos[0].page_content
        video_title = get_video_title(url_video)
        return transcript, video_title
    except Exception as e:
        raise RuntimeError(f"Error loading video info: {e}")


# Hugging Face model setup
def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model,
            temperature=temperature,
            max_new_tokens=250,
            return_full_text=False,
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Error initializing Hugging Face model: {e}")


# Define prompt chain
def llm_chain(model_class):
    system_prompt = "You are a helpful virtual assistant answering a query based on a video transcript, which will be provided below."

    inputs = "Query: {query} \n Transcription: {transcript}"

    if model_class.startswith("hf"):
        user_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{inputs}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = inputs

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    chain = prompt_template | llm | StrOutputParser()

    return chain


# Interpret video content and display results
def interpret_video(url, query="summarize", model_class="hf_hub", language="en", fallback_language="en-GB", translation=None):
    try:
        transcript, video_title = get_video_info(url, language, fallback_language, translation)

        if not transcript:
            print("No transcript available for this video.")
            return

        print(f"## Video Info:\nTitle: {video_title}\nURL: {url}\n")
        chain = llm_chain(model_class)

        print("\n## What is the video about?")
        res = chain.invoke({"transcript": transcript, "query": f"Explain in 1 sentence what this video is about in {language}"})
        print(res)

        print("\n## Main Topics?")
        res = chain.invoke({"transcript": transcript, "query": f"List the main topics of the video in {language}"})
        print(res)

        print("\n## Response to Query?")
        res = chain.invoke({"transcript": transcript, "query": query})
        print(res)

    except RuntimeError as e:
        print(e)
    except Exception as e:
        print("Unexpected error:", e)


# Save the transcript to a file
def save_transcript(transcript, video_title, video_url):
    try:
        video_infos = f"""Video Info:

Title: {video_title}
URL: {video_url}

Transcription:
{transcript}
"""
        with io.open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(video_infos)
        print("Transcript saved to 'transcript.txt'")
    except Exception as e:
        print(f"Error saving transcript: {e}")


# Main program
if __name__ == "__main__":
    # Parameters
    url_video = input("Enter the YouTube URL: ")
    query_user = input("Enter your query: ")
    model_class = "hf_hub"  # Using Hugging Face by default
    language = "en"  # Default language
    fallback_language = "en-GB"  # Fallback language for transcripts

    # Initialize the Hugging Face model
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if not HUGGINGFACE_API_KEY:
        raise ValueError("Hugging Face API key not found. Please set it in your environment variables.")

    llm = model_hf_hub()

    # Process the video and query
    interpret_video(url_video, query_user, model_class, language, fallback_language)
