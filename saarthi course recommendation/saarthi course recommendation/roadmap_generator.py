import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_roadmap(skills_input, df, vectorizer):
    # Split and prioritize skills
    skills = [skill.strip().lower() for skill in skills_input.split(",")]
    skill_weights = {skill: len(skills) - i for i, skill in enumerate(skills)}

    # Vectorize skills and calculate relevance scores
    df['Relevance'] = df['Skills Covered'].apply(lambda x: sum(skill_weights.get(skill, 0) for skill in x.split(", ")))
    df = df.sort_values(by='Relevance', ascending=False)

    # Select courses for the roadmap
    beginner_courses = df[df['Level'] == 'beginner'].head(2)
    intermediate_courses = df[df['Level'] == 'intermediate'].head(2)
    advanced_courses = df[df['Level'] == 'advanced'].head(2)

    return beginner_courses, intermediate_courses, advanced_courses
