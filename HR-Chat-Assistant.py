import os
import gradio as gr
import PyPDF2
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import anthropic  # Import the anthropic module
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('ANTHROPIC_AI_KEY')


# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Check if the file exists in the current directory
file_path = '/workspaces/HR-Chat-Assistant-2/course end project 1 - the_nestle_hr_policy_pdf_2012.pdf'

if not os.path.exists(file_path):
    print(f"The file at path {file_path} does not exist.")
else:
    print(f"The file at path {file_path} exists.")

def load_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Load the HR policy text
hr_policy_text = load_pdf(file_path)
print(hr_policy_text[:500])  # Print the first 500 characters to verify

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to split text into chunks
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Split the HR policy text into chunks
chunks = chunk_text(hr_policy_text)

# Create embeddings for each chunk
embeddings = model.encode(chunks)

# Convert embeddings to numpy array
vector_array = np.array(embeddings).astype('float32')

# Create FAISS index
d = vector_array.shape[1]  # Dimension of vectors
index = faiss.IndexFlatL2(d)
index.add(vector_array)

print("Embeddings created and stored successfully!")

# Function to find the most relevant chunk
def find_most_relevant_chunk(query):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, 1)  # Search for the closest vector
    return chunks[indices[0][0]]

# Function to answer user query with Anthropic Claude
def answer_query(query):
    try:
        most_relevant_chunk = find_most_relevant_chunk(query)
        prompt = f"Context: {most_relevant_chunk}\n\nUser: {query}\nAssistant:"

        # Make the HTTP request to the Anthropic Claude API
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # Debug prints
        print("Sending request to API...")
        print("Endpoint: https://api.anthropic.com/v1/messages")
        print("Headers:", headers)
        print("Data:", data)

        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)

        print("Response Status Code:", response.status_code)
        print("Response Text:", response.text)

        if response.status_code == 200:
            response_json = response.json()
            if 'content' in response_json and isinstance(response_json['content'], list):
                return response_json['content'][0]['text'].strip()
            else:
                return f"Unexpected response format: {response_json}"
        else:
            return f"An error occurred: {response.status_code} {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=answer_query,
    inputs="text",
    outputs="text",
    title="Nestlé HR Chatbot",
    description="Ask any questions about Nestlé's HR policies and get accurate answers."
)

# Launch the interface
iface.launch()
