import pandas as pd
from huggingface_hub import InferenceClient
from llama_index.core import Document, GPTVectorStoreIndex
from scipy import spatial
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()  
api_key = os.getenv("API_KEY")

# Initialize the Inference client for embeddings
client = InferenceClient(api_key=api_key)

# Load your datasets
resumes_df = pd.read_csv('resumes/cleaned_resume_data.csv')  # Contains job category and cleaned resume
categories_df = pd.read_csv('resumes/unique_job_categories.csv')  # Contains distinct job categories

# Step 1: Create a Document Structure
documents = [
    Document(text=row['Cleaned_Resume'], metadata={'category': row['Job_Category']})
    for _, row in resumes_df.iterrows()
]

# Step 2: Create the Index
index = GPTVectorStoreIndex(documents)

# Function to get embeddings
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        response = client.text_embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Generate embeddings for all cleaned resumes
resumes_embeddings = get_embeddings(resumes_df['Cleaned_Resume'].tolist())

# Store embeddings in the documents
for i, doc in enumerate(documents):
    doc.embedding = resumes_embeddings[i]

# Step 3: Define functions for classification and scoring
def classify_and_score_resume(new_resume):
    # Generate embedding for the new resume
    new_embedding = get_embeddings([new_resume])[0]

    # Retrieve the closest matches in the index
    results = index.search(new_embedding, top_k=5)

    # Score based on cosine similarity
    scores = {}
    for result in results:
        category = result.metadata['category']
        cosine_similarity = 1 - spatial.distance.cosine(new_embedding, result.embedding)
        scores[category] = cosine_similarity

    return scores

def classify_resume_with_llama3(resume):
    messages = [
        {"role": "user", "content": f"Classify this resume: {resume}"}
    ]

    stream = client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct", 
        messages=messages, 
        max_tokens=1000,
        stream=True
    )

    classification_result = ""
    for chunk in stream:
        classification_result += chunk.choices[0].delta.content

    return classification_result

# Full processing function for new resumes
def process_resume(new_resume):
    # Classify using Llama3
    job_category = classify_resume_with_llama3(new_resume)
    
    # Retrieve and score based on the index
    scores = classify_and_score_resume(new_resume)

    return job_category, scores

# Example usage
if __name__ == "__main__":
    new_resume = "Education Details May 2013 to May 2017 B.E UIT-RGPV Data Scientist Data Scientist - Matelabs Skill Details Python- Exprience - Less than 1 year months Statsmodels- Exprience - 12 months AWS- Exprience - Less than 1 year months Machine learning- Exprience - Less than 1 year months Sklearn- Exprience - Less than 1 year months Scipy- Exprience - Less than 1 year months Keras- Exprience - Less than 1 year monthsCompany Details company - Matelabs description - ML Platform for business professionals, dummies and enthusiasts. 60/A Koramangala 5th block, Achievements/Tasks behind sukh sagar, Bengaluru, India Developed and deployed auto preprocessing steps of machine learning mainly missing value treatment, outlier detection, encoding, scaling, feature selection and dimensionality reduction. Deployed automated classification and regression model. linkedin.com/in/aditya-rathore- b4600b146 Reasearch and deployed the time series forecasting model ARIMA, SARIMAX, Holt-winter and Prophet. Worked on meta-feature extracting problem. github.com/rathorology Implemented a state of the art research paper on outlier detection for mixed attributes. company - Matelabs description -"
    category, scores = process_resume(new_resume)
    print(f"Classified Job Category: {category}")
    print("Scores for job categories:", scores)
