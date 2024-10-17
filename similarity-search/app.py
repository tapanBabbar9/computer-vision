import streamlit as st
import pandas as pd
import numpy as np
import re
import requests
from io import BytesIO
from PIL import Image
import faiss

# Function to clean up the feature strings
def clean_feature_string(feature_str):
    cleaned_str = re.sub(r'[\[\]]', '', feature_str)  # Remove brackets
    cleaned_values = np.fromstring(cleaned_str, sep=' ')  # Parse values into numpy array
    return cleaned_values

# Function to get top K similar countries using FAISS
def get_top_k_similar_countries(input_country, df, k=5):
    countries = df['Country'].values
    features = np.array([clean_feature_string(f) for f in df['features'].values])
    
    # Find the index of the input country
    try:
        input_idx = list(countries).index(input_country)
    except ValueError:
        return f"Country '{input_country}' not found in the dataset."
    
    input_embedding = features[input_idx].reshape(1, -1)

    # Create a FAISS index for similarity search
    dim = features.shape[1]
    index = faiss.IndexFlatL2(dim)  # Use L2 distance (can be changed to IndexFlatIP for cosine similarity)
    
    # Add all features to the FAISS index
    index.add(features)
    
    # Search for the top K most similar countries
    distances, top_k_idx = index.search(input_embedding, k+1)  # k+1 to exclude the country itself
    
    # Return top K countries with their similarity scores
    return [(countries[i], distances[0][j]) for j, i in enumerate(top_k_idx[0]) if i != input_idx]


# Load multiple CSVs for different models
models = {
    'ViT': pd.read_csv('similarity-search/national_flag_embeddings_vit.csv'),
    'EfficientNet': pd.read_csv('similarity-search/national_flag_embeddings_efficientnet.csv'),
    'DINO-v2': pd.read_csv('similarity-search/national_flag_embeddings_DINO-v2.csv'),
    'clip': pd.read_csv('similarity-search/national_flag_embeddings_clip.csv'),
    'blip': pd.read_csv('similarity-search/national_flag_embeddings_blip.csv'),
    'VGG16': pd.read_csv('similarity-search/national_flag_embeddings_vgg16.csv'),
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
    # Use .read() to avoid the UnidentifiedImageError
    try:
        image_bytes = BytesIO(response.content).read()  # Read the content into bytes
        img = Image.open(BytesIO(image_bytes))  # Open the image from the bytes
        st.image(img, width=200, caption=input_country, use_column_width=False)
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Comparison section
compare_cols = st.columns(len(models))

# Display top 5 similar flags for each model
for model_name, model_df in models.items():
    top_5_countries = get_top_k_similar_countries(input_country, model_df, k=5)

    st.text(f"{model_name} model:")
    cols = st.columns(5)

    for idx, (country, score) in enumerate(top_5_countries):
        country_row = df[df['Country'] == country].iloc[0]
        response = requests.get(country_row['Flag Image'])

        # Use .read() to avoid the UnidentifiedImageError
        try:
            image_bytes = BytesIO(response.content).read()  # Read the content into bytes
            img = Image.open(BytesIO(image_bytes))  # Open the image from the bytes
            with cols[idx % 5]:
                st.image(img, width=100, caption=f"{country}: {score:.4f}")
        except Exception as e:
            st.error(f"Error loading image for {country}: {e}")
            
