import streamlit as st
import cv2
import numpy as np
import pandas as pd
import faiss
import os
import json

# ----------------------------- GLOBAL VARIABLES ----------------------------- #
faiss_index_path = "faiss_index_vgg.bin"
metadata_path = "photo_ids_vgg.json"
image_folder = "drive"
cropped_folder = "cropped_faces_vgg"
distance_thresh = 1

# ----------------------------- HELPER FUNCTIONS ----------------------------- #
def load_faiss_index_and_metadata():
    """Load FAISS index and metadata."""
    try:
        index = faiss.read_index(faiss_index_path)
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        st.success("Loaded FAISS index and metadata successfully.")
        return index, metadata
    except Exception as e:
        st.error(f"Failed to load FAISS index or metadata: {e}")
        return None, None


def search_similar_faces(index, face_embedding, metadata, top_k=5):
    """Search for similar faces in the FAISS index."""
    distances, indices = index.search(np.expand_dims(face_embedding, axis=0), top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(metadata):  # Ensure the index is valid
            result = metadata[idx]
            results.append((result['cropped_face'], dist))
    return results


# ----------------------------- STREAMLIT APPLICATION ----------------------------- #
# Title
st.title("🔍 Face Search Application")

# Load FAISS index and metadata
index, metadata = load_faiss_index_and_metadata()
if index is not None and metadata is not None:
    # Display stored thumbnails
    st.header("📸 Stored Thumbnails")
    cols = st.columns(10)  # Display 5 images per row

    for idx, meta in enumerate(metadata):
        with cols[idx % 10]:
            img_path = os.path.join(cropped_folder, meta['cropped_face'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"Face {idx}", use_column_width=True)

                # Button to search for similar faces
                if st.button(f"Find", key=f"btn_{idx}"):
                    # Use precomputed embedding from metadata
                    embedding = np.array(meta['embedding'], dtype=np.float32)
                    matching_images = search_similar_faces(index, embedding, metadata, top_k=5)
                    st.session_state['matching_images'] = matching_images

    # Display matching images if a search was performed
    if 'matching_images' in st.session_state:
        st.subheader("🎯 Matching Images")
        
        # Loop through the matching images stored in session state
        for img_path, distance in st.session_state['matching_images']:
            if distance < distance_thresh: 
                # Find the metadata entry for this cropped face
                matched_metadata = None
                for meta in metadata:
                    if meta['cropped_face'] == img_path:
                        matched_metadata = meta
                        break

                if matched_metadata:
                    # Get the original image corresponding to the cropped face
                    original_image_path = matched_metadata['original_image']
                    
                    # Load the cropped face image
                    cropped_img_path = os.path.join(cropped_folder, img_path)
                    cropped_img = cv2.imread(cropped_img_path)
                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    # Now load the full image from the image folder
                    img_path_full = os.path.join(image_folder, original_image_path)
                    if os.path.exists(img_path_full):
                        img = cv2.imread(img_path_full)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Create columns for layout
                        cols = st.columns([1, 4])  # 1 for cropped image, 4 for original image

                        # Display the cropped image in the first column
                        with cols[0]:
                            st.image(cropped_img_rgb, caption=f"Cropped Face (Distance: {distance:.4f})", use_column_width=True)

                        # Display the original image in the second column
                        with cols[1]:
                            st.image(img_rgb, caption=f"Original Image", use_column_width=True)
                    else:
                        st.warning(f"Original image not found: {original_image_path}")
                else:
                    st.warning(f"Metadata for cropped face {img_path} not found.")


else:
    st.warning("Please ensure the FAISS index and metadata are available to proceed.")
