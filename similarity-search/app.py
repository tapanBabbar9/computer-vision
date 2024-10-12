import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
from PIL import Image

# Function to clean up the feature strings
def clean_feature_string(feature_str):
    # Remove outer square brackets and split by spaces or commas
    cleaned_str = re.sub(r'[\[\]]', '', feature_str)  # Remove brackets
    cleaned_values = np.fromstring(cleaned_str, sep=' ')  # Parse values into numpy array
    return cleaned_values

# Function to get top 5 similar countries
def get_top_5_similar_countries(input_country, df):
    # Extract country names
    countries = df['Country'].values
    
    # Clean and convert features into arrays
    features = np.array([clean_feature_string(f) for f in df['features'].values])
    
    # Find the index of the input country
    try:
        input_idx = list(countries).index(input_country)
    except ValueError:
        return f"Country '{input_country}' not found in the dataset."
    
    # Get the embedding of the input country
    input_embedding = features[input_idx].reshape(1, -1)
    
    # Compute cosine similarity
    similarities = cosine_similarity(input_embedding, features)[0]
    
    # Get top 5 similar countries
    top_5_idx = similarities.argsort()[-6:-1][::-1]
    
    # Return top 5 countries with similarity scores
    return [(countries[i], similarities[i]) for i in top_5_idx]

# Load CSV file
csv_file = 'flag-similarity/national_flag_embeddings.csv'
df = pd.read_csv(csv_file)

# Streamlit app UI
st.title("Country Flag Similarity Finder")

# Dropdown for selecting the input country
input_country = st.selectbox("Select a country", df['Country'].unique())

if input_country:
    # Display the input country's flag
    st.subheader(f"Input Country: {input_country}")
    input_country_row = df[df['Country'] == input_country].iloc[0]
    response = requests.get(input_country_row['Flag Image'])
    img = Image.open(BytesIO(response.content))
    st.image(img, width=100, caption=input_country, use_column_width=False)

    # Get top 5 similar countries
    top_5_countries = get_top_5_similar_countries(input_country, df)

    # Display top 5 similar flags
    st.subheader("Top 5 similar countries and their flags:")
    # Create a responsive layout
    cols = st.columns(5)  # 5 columns for desktop

    for idx, (country, score) in enumerate(top_5_countries):
        country_row = df[df['Country'] == country].iloc[0]
        response = requests.get(country_row['Flag Image'])
        img = Image.open(BytesIO(response.content))

        # Display flag in respective column
        with cols[idx % 5]:  # Distribute flags in columns (wrap after 5 flags)
            st.image(img, width=100, caption=f"{country}\nScore: {score:.4f}")
