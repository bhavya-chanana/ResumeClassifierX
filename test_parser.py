import importlib
from pyresparser import ResumeParser
data = ResumeParser('resumes\Resume_Template_by_Anubhav.pdf').get_extracted_data()
print(data)


