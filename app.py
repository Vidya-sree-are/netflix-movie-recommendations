import streamlit as st
import openai  
import pandas as pd

# OpenAI API Key
openai.api_key = "your_openai_key"

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('netflix_titles.csv')

netflix_data = load_data()


def generate_recommendations(genre, title_type, year_range):
    # Filter data based on user preferences
    filtered_data = netflix_data[
        (netflix_data['listed_in'].str.contains(genre, case=False, na=False)) &
        (netflix_data['type'] == title_type) &
        (netflix_data['release_year'].between(year_range[0], year_range[1]))
    ]

    # Create a prompt with the filtered data to get OpenAI recommendations
    movies_list = ", ".join(filtered_data['title'].sample(5).values)
    prompt = f"Recommend 5 similar {title_type}s based on the following list of {genre} titles from Netflix: {movies_list}. Provide a brief description of each.Also fetch the poster of the movie."

    # Call the OpenAI API for recommendations
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use the specified chat model (gpt-4)
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=600,  # Adjust the token limit to control the length of the response
        temperature=0.7,  # Controls randomness
    )
    recommendations = response['choices'][0]['message']['content'].strip()
    return recommendations

# Streamlit App Layout
st.title("Netflix Movie Recommendation System with OpenAI")
st.subheader("Find movie and TV show recommendations based on your preferences!")

# User inputs
genre = st.selectbox("Choose a genre", netflix_data['listed_in'].str.split(',').explode().unique())
title_type = st.radio("Type", ["Movie", "TV Show"])
year_range = st.slider("Select Release Year Range", int(netflix_data['release_year'].min()), int(netflix_data['release_year'].max()), (2010, 2023))

# Button to generate recommendations
if st.button("Get Recommendations"):
    with st.spinner("Finding recommendations..."):
        recommendations = generate_recommendations(genre, title_type, year_range)
        st.write("### Recommended Titles")
        st.write(recommendations)
