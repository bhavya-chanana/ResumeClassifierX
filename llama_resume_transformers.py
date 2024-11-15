from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import torch

class ResumeClassifier:
    def __init__(self, model_name: str, resume_data_path: str):
        """
        Initialize the resume classifier with necessary components
        
        Args:
            model_name: Name of the LLaMA model to use
            resume_data_path: Path to CSV with resume data and categories
        """
        # Initialize LLaMA model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # Alternative: Use pipeline for simpler inference
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load and prepare data
        self.resume_data_df = pd.read_csv(resume_data_path)
        
        # Get unique categories from the Category column
        self.unique_categories = self.resume_data_df['Category'].unique()
        
        # Initialize label encoder for categories
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.unique_categories)
        
        # Create embeddings for job categories
        self.job_category_embeddings = self._create_category_embeddings()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'loss': []
        }
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLaMA model using transformers"""
        # Option 1: Using the pipeline
        response = self.pipe(
            prompt,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            return_full_text=False
        )[0]['generated_text']
        
        # Option 2: Using the model and tokenizer directly
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=500,
        #     do_sample=True,
        #     temperature=0.7,
        #     top_p=0.95
        # )
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response.strip()
    
    # Rest of the class methods remain the same...
    def _create_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for all job categories"""
        category_embeddings = {}
        for category in self.unique_categories:
            embedding = self.embedding_model.encode(str(category), convert_to_tensor=True)
            category_embeddings[category] = embedding
        return category_embeddings
    
    def _calculate_similarity_scores(self, resume_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Calculate cosine similarity between resume and all job categories"""
        similarity_scores = []
        for category, category_embedding in self.job_category_embeddings.items():
            similarity = cosine_similarity(
                resume_embedding.reshape(1, -1),
                category_embedding.reshape(1, -1)
            )[0][0]
            similarity_scores.append((category, similarity))
        return sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    def _get_llm_score(self, resume_text: str, category: str) -> float:
        """Get LLM-based relevance score for resume-category pair"""
        prompt = f"""
        On a scale of 0-100, score how well the following resume matches the job category: {category}
        
        Resume:
        {resume_text}
        
        Provide only the numeric score without any explanation.
        """
        
        try:
            score = float(self._get_llm_response(prompt))
            return min(max(score, 0), 100)
        except ValueError:
            return 0.0

# Example usage
if __name__ == "__main__":
    load_dotenv()
    
    # Initialize classifier
    classifier = ResumeClassifier(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        resume_data_path='resumes/cleaned_resume_data.csv'  # Your CSV file path
    )
    
    # Evaluate model
    evaluation_results = classifier.evaluate_model(test_size=0.2, num_epochs=5)
    
    # Print results
    print("\nOverall Metrics:")
    for metric, value in evaluation_results['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")