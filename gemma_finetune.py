import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, LlamaTokenizer
from torch.utils.data import DataLoader
import torch

# Load the Data - load cleaned_resume_data.csv
data = pd.read_csv(r'resumes\\cleaned_resume_data.csv')

# Load tokenizer and model
# Pass authentication token when loading tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it", use_auth_token=True)
# Pass authentication token when loading model
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_auth_token=True
)

# Prepare data
data["input_ids"] = data["Cleaned_Resume"].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512))

# Create dataset
class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

train_dataset = ResumeDataset(data["input_ids"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collator=data_collator)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop - 3 epochs
model.train()
for epoch in range(3):  # Number of epochs
    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save the model
model.save_pretrained("./my_finetuned_model")
