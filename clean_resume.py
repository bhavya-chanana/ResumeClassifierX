import pandas as pd
import unicodedata
import re
'''
creates resumes/cleaned_resume_data.csv after processing and cleaning UpdatedResumeDataSet.csv
'''
# Define the path to your CSV file
input_csv_path = r'resumes\\UpdatedResumeDataSet.csv'
output_csv_path = r'resumes\\cleaned_resume_data.csv'

# Read the CSV file with an encoding that correctly handles special characters (assuming UTF-8)
data = pd.read_csv(input_csv_path, encoding='utf-8')

# Function to clean the resume text
def clean_resume_text(text):
    # Normalize unicode characters to the closest ASCII representation
    text = unicodedata.normalize('NFKD', text)
    # Replace non-ASCII characters with an empty string
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace common encoding errors with an appropriate character or an empty string
    text = text.replace('ÃƒÂ¯', 'i').replace('Ã¢Â€Â¢', '-').replace('Ã', '')
    # Remove excessive white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply the cleaning function to the 'Resume' column
data['Cleaned_Resume'] = data['Resume'].apply(clean_resume_text)

# Ensure that we include the 'Category' column in the output
data.to_csv(output_csv_path, index=False, columns=['Category', 'Cleaned_Resume'], encoding='utf-8-sig')
