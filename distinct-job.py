import pandas as pd

# Load the dataset
file_path = 'resumes/cleaned_resume_data.csv'
resume_data = pd.read_csv(file_path)

# Extract unique job categories
unique_job_categories = resume_data['Category'].unique()

# Convert to DataFrame and save as CSV
unique_job_categories_df = pd.DataFrame(unique_job_categories, columns=['Category'])
output_path = 'resumes/unique_job_categories.csv'
unique_job_categories_df.to_csv(output_path, index=False)

print(f"Unique job categories saved to {output_path}")
