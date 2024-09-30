import pandas as pd
import ast
import streamlit as st

# Load the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv('flag-similarity/national_flag_embeddings.csv')
    # Convert the string representation of embeddings to lists
    print(df['features'])
    df['features'] = df['features'].apply(ast.literal_eval)
    return df

# Load the data
data = load_data()

# Set the title of the Streamlit app
st.title("Country Flag Similarity Search")

# Dropdown for selecting a country
country = st.selectbox("Select a Country", data['country'].tolist())

# Get the embedding for the selected country
selected_country_data = data[data['country'] == country]
selected_embedding = selected_country_data['features'].values[0]

# Function to compute similarity (replace this with your actual similarity function)
def compute_similarity(selected_embedding, all_embeddings):
    # Placeholder for your similarity calculation logic
    # Return indices of most similar flags for example
    # Here, we'll just return the first three for illustration
    return range(len(all_embeddings))

# Compute similarities with all embeddings
similarity_indices = compute_similarity(selected_embedding, data['features'].tolist())

# Display similar flags
st.subheader("Similar Flags:")
for index in similarity_indices:
    similar_country = data.iloc[index]
    st.image(similar_country['flag_image'], caption=similar_country['country'])
