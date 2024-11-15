# llama_resume_classifier_v3.py

from transformers import pipeline
from huggingface_hub import InferenceClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import List, Dict
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMResumeClassifier:
    def __init__(self, resume_data_path: str, cache_dir: str = "cache"):
        """Initialize the LLM-based resume classifier with GPU support"""
        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            torch_dtype=torch.float16,  # Use half precision for GPU memory efficiency
            device_map="auto"  # Automatically handle model placement
        )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.knowledge_base_path = self.cache_dir / "knowledge_base.json"
        self.plots_dir = self.cache_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load and prepare data
        self.resume_data = pd.read_csv(resume_data_path)
        
        # Split data into train/test (70/30)
        self.train_data, self.test_data = train_test_split(
            self.resume_data, 
            test_size=0.3, 
            random_state=42,
            stratify=self.resume_data['Category']
        )
        
        self.category_knowledge_base = self._load_or_create_knowledge_base()
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

    def _create_knowledge_base(self) -> str:
        """Create enhanced knowledge base with skills and experience patterns"""
        knowledge_base = "Job Categories with Required Skills and Experience Patterns:\n\n"
        
        for category in self.train_data['Category'].unique():
            # Get category examples
            examples = self.train_data[self.train_data['Category'] == category]['Cleaned_Resume'].head(3)
            
            # Extract common skills and patterns
            skills = set()
            for resume in examples:
                # Extract skills using regex patterns
                found_skills = re.findall(r'(?i)(python|java|sql|machine learning|deep learning|tableau|analytics|r|spark|hadoop|aws|azure|nlp|ai|statistics|visualization)', str(resume))
                skills.update(found_skills)
            
            knowledge_base += f"\nCategory: {category}\n"
            knowledge_base += "Typical Skills:\n"
            knowledge_base += ", ".join(sorted(skills)) + "\n"
            knowledge_base += "Example Resumes:\n"
            
            # Add summarized examples
            for idx, example in enumerate(examples, 1):
                # Get first 100 words as summary
                summary = " ".join(str(example).split()[:100])
                knowledge_base += f"Example {idx}:\n{summary}...\n"
                
        return knowledge_base

    def _load_or_create_knowledge_base(self) -> str:
        """Load existing knowledge base or create new one"""
        if self.knowledge_base_path.exists():
            with open(self.knowledge_base_path, 'r') as f:
                saved_data = json.load(f)
                return saved_data['knowledge_base']
                
        knowledge_base = self._create_knowledge_base()
        self._save_knowledge_base(knowledge_base)
        return knowledge_base

    def _save_knowledge_base(self, knowledge_base: str):
        """Save knowledge base with timestamp"""
        data = {
            'knowledge_base': knowledge_base,
            'created_at': datetime.now().isoformat(),
            'categories': self.train_data['Category'].unique().tolist()
        }
        with open(self.knowledge_base_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_llm_classification(self, resume_text: str) -> str:
        """Get classification using GPU-accelerated model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                resume_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate with error handling
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result.strip()
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return None

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model with proper error handling"""
        predictions = []
        actuals = []
        
        print("\nEvaluating model on test set...")
        for _, row in tqdm(self.test_data.iterrows(), total=len(self.test_data)):
            try:
                resume_text = str(row['Cleaned_Resume'])
                true_category = row['Category']
                
                pred_category = self._get_llm_classification(resume_text)
                if pred_category is not None:
                    predictions.append(pred_category)
                    actuals.append(true_category)
                    
            except Exception as e:
                print(f"Error processing resume: {e}")
                continue

        # Check if we have predictions
        if not predictions:
            return {
                'error': 'No valid predictions generated',
                'accuracy': 0.0,
                'classification_report': None
            }

        try:
            metrics = {
                'accuracy': accuracy_score(actuals, predictions),
                'classification_report': classification_report(
                    actuals, 
                    predictions,
                    zero_division=0
                )
            }
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {
                'error': str(e),
                'accuracy': 0.0,
                'classification_report': None
            }

    def plot_category_distribution(self):
        """Enhanced category distribution plot with skills breakdown"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Category distribution
        sns.countplot(data=self.train_data, y='Category', ax=ax1)
        ax1.set_title('Distribution of Resume Categories')
        
        # Skills distribution per category 
        skills_by_category = {}
        common_skills = ['Python', 'SQL', 'Machine Learning', 'Java', 'Analytics']
        
        for category in self.train_data['Category'].unique():
            cat_resumes = self.train_data[self.train_data['Category'] == category]['Cleaned_Resume']
            skills_count = {skill: sum(1 for r in cat_resumes if skill.lower() in str(r).lower()) 
                           for skill in common_skills}
            skills_by_category[category] = skills_count
        
        # Plot skills heatmap
        skills_df = pd.DataFrame(skills_by_category).T
        sns.heatmap(skills_df, annot=True, cmap='YlOrRd', ax=ax2)
        ax2.set_title('Skills Distribution by Category')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'category_analysis.png')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, categories):
        """Plot confusion matrix of predictions"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories,
                   yticklabels=categories)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png')
        plt.close()

    def plot_metrics_history(self):
        """Plot metrics history"""
        plt.figure(figsize=(10, 6))
        for metric in self.metrics:
            if metric != 'confusion_matrix':
                plt.plot(self.metrics[metric], label=metric)
        plt.title('Classification Metrics Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'metrics_history.png')
        plt.close()

if __name__ == "__main__":
    # Initialize classifier (no API key needed)
    classifier = LLMResumeClassifier(
        resume_data_path='resumes\cleaned_resume_data.csv'
    )
    
    # Evaluate model
    results = classifier.evaluate_model()
    
    print("\nClassification Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nDetailed Classification Report:")
    print(results['classification_report'])
    print("\nVisualization plots saved in: {classifier.plots_dir}")