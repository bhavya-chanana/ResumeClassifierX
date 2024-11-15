from huggingface_hub import InferenceClient
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

class ResumeClassifier:
    def __init__(self, api_key: str, resume_data_path: str):
        """
        Initialize the resume classifier with necessary components
        
        Args:
            api_key: HuggingFace API key
            resume_data_path: Path to CSV with resume data and categories
        """
        self.client = InferenceClient(api_key=api_key)
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
        
    def _create_category_embeddings(self) -> Dict[str, np.ndarray]:
        """Create embeddings for all job categories"""
        category_embeddings = {}
        for category in self.unique_categories:
            embedding = self.embedding_model.encode(str(category), convert_to_tensor=True)
            category_embeddings[category] = embedding
        return category_embeddings
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get response from Llama model"""
        messages = [{"role": "user", "content": prompt}]
        stream = self.client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=True
        )
        
        response = ""
        for chunk in stream:
            response += chunk.choices[0].delta.content
        return response
    
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
    
    def _calculate_cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate cross entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred))
    
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
    
    def classify_resume(self, resume_text: str, top_k: int = 3) -> List[Dict]:
        """Classify a resume and return top k matching categories with scores"""
        resume_embedding = self.embedding_model.encode(resume_text, convert_to_tensor=True)
        similarity_scores = self._calculate_similarity_scores(resume_embedding)
        top_categories = similarity_scores[:top_k]
        
        results = []
        for category, embedding_similarity in top_categories:
            llm_score = self._get_llm_score(resume_text, category)
            results.append({
                'category': category,
                'embedding_similarity': float(embedding_similarity),
                'llm_score': llm_score,
                'combined_score': (float(embedding_similarity) + llm_score) / 2
            })
        
        return sorted(results, key=lambda x: x['combined_score'], reverse=True)
    
    def evaluate_model(self, test_size: float = 0.2, num_epochs: int = 3) -> Dict[str, float]:
        """
        Evaluate model performance on a test split of the data over multiple epochs.

        Args:
            test_size: Fraction of data to use for testing
            num_epochs: Number of epochs to run for evaluation

        Returns:
            Dictionary containing evaluation metrics
        """
        from sklearn.model_selection import train_test_split
        
        # Split data into train and test
        test_data = self.resume_data_df.sample(frac=test_size, random_state=42)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            y_true = []
            y_pred = []
            y_pred_proba = []

            # Process test data
            for _, row in tqdm(test_data.iterrows(), desc="Evaluating", total=len(test_data)):
                true_category = row['Category']
                resume_text = row['Cleaned_Resume']
                
                # Skip if resume text is empty or NaN
                if pd.isna(resume_text) or str(resume_text).strip() == '':
                    continue
                    
                predictions = self.classify_resume(str(resume_text), top_k=1)
                pred_category = predictions[0]['category']
                pred_score = predictions[0]['combined_score'] / 100  # Normalize to 0-1
                
                y_true.append(self.label_encoder.transform([true_category])[0])
                y_pred.append(self.label_encoder.transform([pred_category])[0])
                y_pred_proba.append(pred_score)

            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_pred_proba = np.array(y_pred_proba)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'loss': self._calculate_cross_entropy_loss(
                    pd.get_dummies(y_true).values,
                    np.eye(len(self.label_encoder.classes_))[y_pred]
                )
            }
            
            # Update metrics history
            for metric, value in metrics.items():
                if metric not in self.metrics_history:
                    self.metrics_history[metric] = []  # Initialize the list if it doesn't exist
                self.metrics_history[metric].append(value)
            
            # Print overall metrics for the epoch
            print("\nOverall Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
        return {
            'overall_metrics': metrics,
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

# Example usage
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("API_KEY")
    
    # Initialize classifier
    classifier = ResumeClassifier(
        api_key=api_key,
        resume_data_path='resumes/cleaned_resume_data.csv'  # Your CSV file path
    )
    
    # Evaluate model
    evaluation_results = classifier.evaluate_model(test_size=0.2, num_epochs=3)

    
    # Print results
    print("\nOverall Metrics:")
    for metric, value in evaluation_results['overall_metrics'].items():
        print(f"{metric}: {value:.4f}")
    
    # Plot metrics history
    classifier.plot_metrics_history()
