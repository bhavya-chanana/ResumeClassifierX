import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_path = './my_finetuned_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Prepare input
parsed_resume_text = "Your parsed resume text here."
input_ids = tokenizer(parsed_resume_text, return_tensors='pt', truncation=True, max_length=512).input_ids

# Generate or evaluate
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits
    # Interpret your logits here according to the specifics of your task
    # For example, extract a score, classify, or generate text

# Display results
print("Model output:", predictions)
