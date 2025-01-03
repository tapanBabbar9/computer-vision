import streamlit as st
import cv2
import numpy as np
import pandas as pd
import faiss
import os
import json

# ----------------------------- GLOBAL VARIABLES ----------------------------- #
base_path = os.path.dirname(os.path.abspath(__file__))
faiss_index_path = os.path.join(base_path, "oscar", "faiss_index.bin")
metadata_path = os.path.join(base_path, "oscar", "photo_ids.json")
image_folder = "oscar/photos"
cropped_folder = "oscar/cropped_faces"
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



def search_similar_faces(index, face_embedding, metadata, top_k=100):
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
st.caption("This project showcases the use of YuNet, VGG-16, and FAISS for efficient photo recognition and reverse image search. All images used belong to 'Academy of Motion Picture Arts and Sciences' and are used for educational purposes")

# Load FAISS index and metadata
index, metadata = load_faiss_index_and_metadata()
if index is not None and metadata is not None:
    # Display stored thumbnails
    st.header("📸 Stored Thumbnails")
    cols = st.columns(10)  # Display 5 images per row
    whitelist = ["9_face_0.jpg",
                 "9_face_1.jpg",
                 "9_face_2.jpg",
                 "9_face_7.jpg",
                 "9_face_4.jpg",
                 ]  
    # Filter metadata based on the whitelist
    filtered_metadata = [meta for meta in metadata if meta['cropped_face'] in whitelist]

    # Calculate the number of columns based on the filtered metadata
    num_columns = min(len(filtered_metadata), 10)  # Adjust number of columns (e.g., 3 columns max)
    cols = st.columns(num_columns)

    for idx, meta in enumerate(filtered_metadata):
        col_idx = idx % num_columns  # Ensure each image is assigned to the correct column
        with cols[col_idx]:
            img_path = os.path.join(base_path, cropped_folder, meta['cropped_face'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"Face {idx}")

                # Button to search for similar faces
                if st.button(f"Find", key=f"btn_{idx}"):
                    # Use precomputed embedding from metadata
                    embedding = np.array(meta['embedding'], dtype=np.float32)
                    matching_images = search_similar_faces(index, embedding, metadata, top_k=100)
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
                    cropped_img_path = os.path.join(base_path, cropped_folder, img_path)
                    cropped_img = cv2.imread(cropped_img_path)
                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    # Now load the full image from the image folder
                    img_path_full = os.path.join(base_path, image_folder, original_image_path)
                    if os.path.exists(img_path_full):
                        img = cv2.imread(img_path_full)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Create columns for layout
                        cols = st.columns([1, 9])  # 1 for cropped image, 4 for original image

                        # Display the cropped image in the first column
                        with cols[0]:
                            #st.image(cropped_img_rgb, caption=f"Cropped Face (Distance: {distance:.4f})")
                            st.image(cropped_img_rgb)

                        # Display the original image in the second column
                        with cols[1]:
                            st.image(img_rgb)
                    else:
                        st.warning(f"Original image not found: {original_image_path}")
                else:
                    st.warning(f"Metadata for cropped face {img_path} not found.")


else:
    st.warning("Please ensure the FAISS index and metadata are available to proceed.")
