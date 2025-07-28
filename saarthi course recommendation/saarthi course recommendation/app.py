import streamlit as st
import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
from roadmap_generator import generate_roadmap  # Import the roadmap generator

# Load model and encoders
with open("model.pkl", "rb") as f:
    model, vectorizer, type_encoder, level_encoder, df = pickle.load(f)

# Function to fetch link preview
def fetch_link_preview(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string if soup.title else "No Title Available"
        description = soup.find("meta", attrs={"name": "description"})
        description = description["content"] if description else "No Description Available"
        return title, description
    except Exception:
        return "No Title Available", "No Description Available"

# Function to create a clickable button for a link
def create_link_button(label, url):
    return f"""
    <a href="{url}" target="_blank" style="
        display: inline-block;
        padding: 10px 15px;
        font-size: 16px;
        color: white;
        background-color: #007BFF;
        text-decoration: none;
        border-radius: 5px;
        text-align: center;
    ">{label}</a>
    """

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Saarthi - AI Course Recommendation System</h1>", unsafe_allow_html=True)

# User inputs
name = st.text_input("Name:", "")
st.markdown("<h2>Select Course Type</h2>", unsafe_allow_html=True)
course_type = st.selectbox("", type_encoder.classes_)
st.markdown("<h2>Select Course Level</h2>", unsafe_allow_html=True)
course_level = st.selectbox("", level_encoder.classes_)
st.markdown("<h2>Enter Skills (comma-separated, e.g., Python, SQL, Machine Learning)</h2>", unsafe_allow_html=True)
skills_input = st.text_area("")

if st.button("Recommend Courses"):
    # Preprocess inputs
    skills_vector = vectorizer.transform([skills_input.lower()])
    type_encoded = type_encoder.transform([course_type])[0]
    level_encoded = level_encoder.transform([course_level])[0]
    user_features = pd.concat([
        pd.DataFrame(skills_vector.toarray()),
        pd.DataFrame([[type_encoded, level_encoded]])
    ], axis=1)

    # Get recommendations
    distances, indices = model.kneighbors(user_features)
    recommendations = df.iloc[indices[0]].head(10)

    # Display recommendations
    st.markdown(f"<h2>Hi {name}, these are the Courses tailored Just for you ⬇️:</h2>", unsafe_allow_html=True)
    for _, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"<h3>{row['Title']}</h3>", unsafe_allow_html=True)
            st.write(f"**Course Type**: {row['Type'].capitalize()}")
            st.write(f"**Course Duration**: {row['Duration']}")
            st.write(f"**Skills Covered**: {row['Skills Covered']}")
            title, description = fetch_link_preview(row['URL'])
            st.write(f"**Preview**: {title} - {description}")
            st.markdown(create_link_button(f"Go to {row['Title']}", row['URL']), unsafe_allow_html=True)
            st.write("---")

    # Generate personalized roadmap
    beginner_courses, intermediate_courses, advanced_courses = generate_roadmap(skills_input, df, vectorizer)

    st.markdown(f"<h2>Your Personalized Roadmap, {name} 🛣️:</h2>", unsafe_allow_html=True)

    # Beginner courses dropdown
    with st.expander("Beginner Courses"):
        for _, row in beginner_courses.iterrows():
            with st.container():
                st.markdown(f"<h4>{row['Title']}</h4>", unsafe_allow_html=True)
                st.write(f"**Course Type**: {row['Type'].capitalize()}")
                st.write(f"**Course Duration**: {row['Duration']}")
                st.write(f"**Skills Covered**: {row['Skills Covered']}")
                title, description = fetch_link_preview(row['URL'])
                st.write(f"**Preview**: {title} - {description}")
                st.markdown(create_link_button(f"Go to {row['Title']}", row['URL']), unsafe_allow_html=True)
                st.write("---")

    # Intermediate courses dropdown
    with st.expander("Intermediate Courses"):
        for _, row in intermediate_courses.iterrows():
            with st.container():
                st.markdown(f"<h4>{row['Title']}</h4>", unsafe_allow_html=True)
                st.write(f"**Course Type**: {row['Type'].capitalize()}")
                st.write(f"**Course Duration**: {row['Duration']}")
                st.write(f"**Skills Covered**: {row['Skills Covered']}")
                title, description = fetch_link_preview(row['URL'])
                st.write(f"**Preview**: {title} - {description}")
                st.markdown(create_link_button(f"Go to {row['Title']}", row['URL']), unsafe_allow_html=True)
                st.write("---")

    # Advanced courses dropdown
    with st.expander("Advanced Courses"):
        for _, row in advanced_courses.iterrows():
            with st.container():
                st.markdown(f"<h4>{row['Title']}</h4>", unsafe_allow_html=True)
                st.write(f"**Course Type**: {row['Type'].capitalize()}")
                st.write(f"**Course Duration**: {row['Duration']}")
                st.write(f"**Skills Covered**: {row['Skills Covered']}")
                title, description = fetch_link_preview(row['URL'])
                st.write(f"**Preview**: {title} - {description}")
                st.markdown(create_link_button(f"Go to {row['Title']}", row['URL']), unsafe_allow_html=True)
                st.write("---")