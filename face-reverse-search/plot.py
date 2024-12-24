import streamlit as st
import cv2
import numpy as np
import faiss
import os
import json
import pandas as pd
from sklearn.cluster import DBSCAN

# ----------------------------- GLOBAL VARIABLES ----------------------------- #
faiss_index_path = "faiss_index_vgg.bin"
metadata_file = "photo_ids_vgg.json"
image_folder = "drive"
faiss_index_csv = "faiss_index.csv"
cropped_folder = "cropped_faces_vgg"
# ----------------------------- HELPER FUNCTIONS ----------------------------- #
def load_faiss_index():
    """Load FAISS index and photo IDs."""
    try:
        # Load FAISS index
        index = faiss.read_index(faiss_index_path)
        # Load photo IDs from the CSV
        photo_ids = pd.read_csv(faiss_index_csv)['photo_ids'].tolist()
        st.success(f"Loaded FAISS index with {len(photo_ids)} faces.")
        return index, photo_ids
    except Exception as e:
        st.error(f"Error loading FAISS index or photo IDs: {e}")
        return None, None

def load_metadata():
    """Load metadata from a saved JSON file."""
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return []

def load_embeddings_from_metadata(metadata):
    """Extract embeddings and photo IDs from metadata."""
    embeddings = [np.array(entry['embedding'], dtype=np.float32) for entry in metadata]
    photo_ids = [entry['cropped_face'] for entry in metadata]
    return embeddings, photo_ids

def cluster_faces(embeddings):
    """Cluster faces using DBSCAN based on embeddings."""
    db = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    return db.fit_predict(embeddings)

# ----------------------------- STREAMLIT APPLICATION ----------------------------- #
st.title("🔍 Face Grouping Application")

# Load metadata and embeddings
metadata = load_metadata()
face_embeddings, photo_ids = load_embeddings_from_metadata(metadata)

# Initialize session state if not already initialized
if 'index' not in st.session_state:
    st.session_state.index, _ = load_faiss_index()

if 'photo_ids' not in st.session_state:
    st.session_state.photo_ids = photo_ids

if 'unique_faces' not in st.session_state:
    st.session_state.unique_faces = []

if 'face_embeddings' not in st.session_state:
    st.session_state.face_embeddings = face_embeddings

# Check if the FAISS index is available
if st.session_state.index is not None:
    # Prepare faces for display
    st.session_state.unique_faces = []
    for photo_id in st.session_state.photo_ids:
        img_path = os.path.join(cropped_folder, photo_id)
        matched_metadata = next((entry for entry in metadata if entry['cropped_face'] == photo_id), None)
        if matched_metadata:
            cropped_face = cv2.imread(img_path)
            st.session_state.unique_faces.append((cropped_face, photo_id))

    # Cluster the faces based on embeddings
    labels = cluster_faces(st.session_state.face_embeddings)
    
    # Group faces by cluster label (i.e., person)
    grouped_faces = {}
    for idx, label in enumerate(labels):
        if label not in grouped_faces:
            grouped_faces[label] = []
        grouped_faces[label].append(st.session_state.unique_faces[idx])

    # Display grouped faces
    st.header("📸 Grouped Faces")

    # Loop through the groups and display them
    for label, faces in grouped_faces.items():
        num_faces = len(faces)
        num_columns = min(10, num_faces)  # Limit to 10 columns at most
        cols = st.columns(num_columns)  # Create columns dynamically

        st.subheader(f"Person {label} Faces:")

        displayed_faces = set()  # Track displayed faces to avoid duplicates
        for idx, (face, photo_id) in enumerate(faces):
            if photo_id not in displayed_faces:
                # Calculate column index and show face in the corresponding column
                with cols[idx % num_columns]:
                    img_path = os.path.join(cropped_folder, photo_id)  # Get the correct path
                    cropped_face = cv2.imread(img_path)
                    
                    if cropped_face is None:
                        st.warning(f"Warning: Failed to load image at {img_path}")
                    else:
                        # Convert to RGB for displaying in Streamlit
                        face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                        st.image(face_rgb, caption=f"Face from {photo_id}", use_column_width=True)
                displayed_faces.add(photo_id)  # Mark as displayed
