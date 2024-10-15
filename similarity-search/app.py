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
    cleaned_str = re.sub(r'[\[\]]', '', feature_str)  # Remove brackets
    cleaned_values = np.fromstring(cleaned_str, sep=' ')  # Parse values into numpy array
    return cleaned_values

# Function to get top K similar countries
def get_top_k_similar_countries(input_country, df, k=5):
    countries = df['Country'].values
    features = np.array([clean_feature_string(f) for f in df['features'].values])
    
    # Find the index of the input country
    try:
        input_idx = list(countries).index(input_country)
    except ValueError:
        return f"Country '{input_country}' not found in the dataset."
    
    input_embedding = features[input_idx].reshape(1, -1)
    similarities = cosine_similarity(input_embedding, features)[0]
    
    top_k_idx = similarities.argsort()[-(k+1):-1][::-1]
    return [(countries[i], similarities[i]) for i in top_k_idx]

# Load multiple CSVs for different models
models = {
    'ViT': pd.read_csv('similarity-search/national_flag_embeddings_vit.csv'),
    'EfficientNet': pd.read_csv('similarity-search/national_flag_embeddings_efficientnet.csv'),
    'VGG16': pd.read_csv('similarity-search/national_flag_embeddings_vgg16.csv'),
    'DINO-v2': pd.read_csv('similarity-search/national_flag_embeddings_DINO-v2.csv'),
    'clip': pd.read_csv('similarity-search/national_flag_embeddings_clip.csv'),
    'blip': pd.read_csv('similarity-search/national_flag_embeddings_blip.csv'),
}

# Streamlit UI
st.title("Country Flag Similarity Finder - Model Comparison")

# Dropdown for selecting the model
selected_model = "VGG16"
df = models[selected_model]
# Dropdown for selecting the input country
input_country = st.selectbox("Select a country", df['Country'].unique())

if input_country:
    st.subheader(f"Input Country: {input_country}")
    input_country_row = df[df['Country'] == input_country].iloc[0]
    response = requests.get(input_country_row['Flag Image'])
    img = Image.open(BytesIO(response.content))
    st.image(img, width=200, caption=input_country, use_column_width=False)

# Comparison section
compare_cols = st.columns(len(models))

# Display top 5 similar flags for each model
for model_name, model_df in models.items():
    top_5_countries = get_top_k_similar_countries(input_country, model_df, k=5)

    st.text(f"Top 5 similar countries - {model_name} model:")
    cols = st.columns(5)

    for idx, (country, score) in enumerate(top_5_countries):
        country_row = df[df['Country'] == country].iloc[0]
        response = requests.get(country_row['Flag Image'])
        img = Image.open(BytesIO(response.content))
        with cols[idx % 5]:
            st.image(img, width=100, caption=f"{country}\nScore: {score:.4f}")
            
