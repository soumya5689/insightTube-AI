"# insightTube-AI" 
"# insightTube-AI" 
#   i n s i g h t T u b e - A I 

[image](https://github.com/user-attachments/assets/4cc6f3e5-4f1b-486c-a74f-49fbcd85cf2b)
 
YouTube Video Transcript Interpreter
This project is a Python-based application that leverages LangChain, Hugging Face models, and BeautifulSoup to extract, process, and analyze YouTube video transcripts. It provides insightful responses to user queries based on the video's content.

Features
YouTube Transcript Extraction: Automatically retrieves the transcript of a YouTube video in the specified language.
Content Interpretation: Answers user queries about the video's content using AI.
Hugging Face Integration: Utilizes Hugging Face models for natural language processing.
Save Transcripts: Saves the video transcript and metadata to a local file.
Query-based Analysis: Generates summaries, lists main topics, and answers user-defined questions.

Requirements
Python 3.8 or higher
Hugging Face API Key
YouTube video URL
Internet connection
Installation
1. Clone the repository
git clone <repository_url>
cd <repository_name>

2. Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt
5. Configure the environment variables
Create a .env file in the project root and add your Hugging Face API key:
HUGGINGFACE_API_KEY=your_huggingface_api_key

1. Run the Application
Execute the main Python script:
python app.py

3. Provide Input
YouTube URL: Paste the URL of the video you want to analyze.
Query: Type your query for the AI assistant to process.

5. Example Queries
"Summarize the video."
"What are the main topics covered?"
"What is the video about?"

7. Output
Video Information: Title and URL of the video.
Summary: A one-sentence explanation of the video's content.
Topics: List of main topics discussed in the video.
Query Response: Detailed answer to your input query.




 
