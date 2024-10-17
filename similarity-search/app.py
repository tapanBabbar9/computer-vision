import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from PIL import Image
import faiss


st.set_page_config(
    page_title="Flag similarity search",
    page_icon="🏴‍☠️",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://github.com/tapanBabbar9/computer-vision",
        'About': "https://medium.com/@tapanbabbar"
    }
)
IMAGE_DIR = "similarity-search/images"

# Load multiple CSVs for different models
models = {
    'ViT': pd.read_csv('similarity-search/embeddings/national_flag_embeddings_vit.csv'),
    'EfficientNet': pd.read_csv('similarity-search/embeddings/national_flag_embeddings_efficientnet.csv'),
    'DINO-v2': pd.read_csv('similarity-search/embeddings/national_flag_embeddings_DINO-v2.csv'),
    'clip': pd.read_csv('similarity-search/embeddings/national_flag_embeddings_clip.csv'),
    'blip': pd.read_csv('similarity-search/embeddings/national_flag_embeddings_blip.csv'),
    'VGG16': pd.read_csv('similarity-search/embeddings/national_flag_embeddings_vgg16.csv'),
}

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

# Function to load an image from the local folder based on the country name
def load_local_image(country_name):
    # Sanitize the country name to match the local image file naming convention
    sanitized_country_name = country_name.replace(" ", "_").replace("[", "").replace("]", "")
    
    # Path to the local image file
    image_path = os.path.join(IMAGE_DIR, f"{sanitized_country_name}.png")

    # Check if the image exists in the folder
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        print(f"Image for {country_name} not found.")
        return None

# Streamlit UI
st.subheader("Top 5 Similar Flags – Transformer Model Comparison")

# Dropdown for selecting the model
selected_model = "VGG16"
df = models[selected_model]
# Dropdown for selecting the input country
input_country = st.selectbox("Select a country", df['Country'].unique())

if input_country:
    st.subheader(f"Input Country: {input_country}")
    
    # Load the input country's image from the local folder
    img = load_local_image(input_country)
    
    # If the image was found, display it
    if img:
        st.image(img, width=200, caption=input_country, use_column_width=False)

# Comparison section
compare_cols = st.columns(len(models))

# Display top 5 similar flags for each model
for model_name, model_df in models.items():
    top_5_countries = get_top_k_similar_countries(input_country, model_df, k=5)

    st.text(f"{model_name} model:")
    cols = st.columns(5)

    for idx, (country, score) in enumerate(top_5_countries):
        # Load the flag image for each country from the local folder
        img = load_local_image(country)

        if img:
            with cols[idx % 5]:
                st.image(img, width=100, caption=f"{country}: {score:.4f}")
        else:
            st.error(f"Image for {country} not found.")
