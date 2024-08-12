def resume_to_text(resume_data):
    """
    Converts parsed resume data into a text string suitable for the model.
    """
    parts = []

    # Combining skills
    if resume_data.get('skills'):
        skills_text = 'Skills: ' + ', '.join(resume_data['skills'])
        parts.append(skills_text)

    # Adding degree
    if resume_data.get('degree'):
        degree_text = 'Education: ' + ' '.join(resume_data['degree']).replace('\n', ', ')
        parts.append(degree_text)

    # Combining experience
    if resume_data.get('experience'):
        experience_text = 'Experience: ' + ' '.join(resume_data['experience'])
        parts.append(experience_text)

    print(' '.join(parts))

    return ' '.join(parts)

resume_data = {'name': 'Govind Anjan', 'email': 'modhasaladi75@gmail.com', 'mobile_number': '7303415156', 'skills': ['Tensorflow', 'Github', 'Operations', 'Reporting', 'Analysis', 'Django', 'Java', 'Sql', 'Budget', 'Operating systems', 'System', 'Networking', 'Python', 'Time management', 'Design', 'Health', 'Email', 'Nltk', 'Postgresql', 'R', 'Sci', 'Opencv', 'Linux', 'Mysql', 'Windows', 'Database', 'Machine learning', 'Algorithms', 'Engineering', 'C', 'Ai', 'Javascript', 'Mobile', 'Travel', 'Pytorch', 'C++', 'Warehouse', 'Keras', 'Docker'], 'college_name': None, 'degree': None, 'designation': ['Data Fusion Developer'], 'experience': ['Python, C++, Java, SQL, C, R, JavaScript', 'TensorFlow, Keras, Django, Sci-kit, Nltk, OpenCV, PyTorch', 'Docker, GIT, PostgreSQL, MySQL, SQLite', 'Linux, Windows, MacOS, Web', 'Leadership, Flexibility, Time Management, Problem Solving', 'Delhi, India', 'Mar 2017 - Mar 2019', 'Vishakapatnam, India', 'Mar 2017', '• FrontRow', 'Product Operations (Internship)', 'Remote', 'Jan 2022 - Jun 2022', '◦ Product Development: Handling product development and operations for FrontRow, which facilitates e-learning for', 'co-curricular activities.', '◦ Handling Activities: Hosting events for students at the company which helps them to improve in areas which they are', 'currently attending.', '• ROSYS Virtual Solutions.', 'Remote', 'Jun 2022 - Jan 2023', '◦ Project Course - To create an application which is based on agriculture: Created project-based course using', 'Data Fusion Developer', 'Deep Learning, Computer Vision, and Machine Learning for recommender system.', '◦ Optimisation: Implemented CNN layers which would be efficient at the same time accurate enough to detect the health', 'status of the crop.', '◦ Warehousing and Database: Using PostgresSQL in order to imitate the warehouse condition and also store the data', 'of all warehouse and needed requirements during the course of farming.', '• Rolls Royce.', 'Data Science Intern', 'Bangalore, India', 'Sep 2023 - Feb 2024'], 'company_names': None, 'no_of_pages': 1, 'total_experience': 3.42}

resume_to_text(resume_data)
