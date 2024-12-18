import streamlit as st
import cv2
import numpy as np
import pandas as pd
import faiss
import os
from deepface import DeepFace

# ----------------------------- GLOBAL VARIABLES ----------------------------- #
yunet = cv2.FaceDetectorYN.create(
    model="face_detection_yunet_2023mar.onnx",
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000
)

facenet_model = "Facenet"
embedding_dimension = 128
faiss_index_file = "faiss_index.csv"
photo_ids = []  # Tracks image filenames


# ----------------------------- HELPER FUNCTIONS ----------------------------- #
def detect_and_crop_faces(image_path, return_boxes=False):
    """ Detect faces in an image and crop them. """
    img = cv2.imread(image_path)
    if img is None:
        return [] if not return_boxes else ([], [])
    
    height, width = img.shape[:2]
    yunet.setInputSize((width, height))
    _, faces = yunet.detect(img)

    cropped_faces, face_boxes = [], []
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cropped_face = img[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)
            face_boxes.append((x, y, w, h))
    
    return (cropped_faces, face_boxes) if return_boxes else cropped_faces


def get_embeddings(face_images):
    """ Generate embeddings for cropped faces. """
    embeddings = []
    for face_img in face_images:
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(img_path=face_rgb, model_name=facenet_model, enforce_detection=False)[0]['embedding']
        embeddings.append(np.array(embedding, dtype=np.float32))
    return embeddings


def process_and_store_images(image_folder):
    """ Process images and store their embeddings into a FAISS index. """
    global photo_ids, index
    index = faiss.IndexFlatL2(embedding_dimension)

    for photo_id, image_file in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        faces = detect_and_crop_faces(image_path)
        if faces:
            embeddings = get_embeddings(faces)
            for embedding in embeddings:
                index.add(np.expand_dims(embedding, axis=0))
                photo_ids.append(image_file)

    # Save FAISS index as CSV
    save_faiss_index()
    st.success(f"Processed and stored {len(photo_ids)} faces.")


def save_faiss_index():
    """ Save FAISS index and image mappings to CSV. """
    faiss.write_index(index, "faiss.index")
    pd.DataFrame({'photo_ids': photo_ids}).to_csv(faiss_index_file, index=False)

def load_faiss_index():
    """Load FAISS index and image mappings from CSV."""
    try:
        index = faiss.read_index("faiss.index")  # Load FAISS index
        photo_ids = pd.read_csv("faiss_index.csv")['photo_ids'].tolist()  # Load photo IDs
        st.success(f"Loaded FAISS index with {len(photo_ids)} faces.")
        return index, photo_ids  # Return both index and photo_ids
    except Exception as e:
        st.error(f"Error loading FAISS index or photo IDs: {e}")
        return None, None


def search_similar_faces(index, face_embedding, top_k=5):
    """Search for similar faces in the FAISS index."""
    distances, indices = index.search(np.expand_dims(face_embedding, axis=0), top_k)
    print("FAISS Search Output:")
    print("Distances:", distances)
    print("Indices:", indices)
    photo_ids = pd.read_csv("faiss_index.csv")['photo_ids'].tolist() 
    # Dictionary to store the closest distance for each unique photo_id
    unique_results = {}

    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(photo_ids):  # Ensure the index is within bounds
            photo_id = photo_ids[idx]
            # If photo_id is not in the dictionary, or a closer match is found, update it
            if photo_id not in unique_results or dist < unique_results[photo_id]:
                unique_results[photo_id] = dist

    # Convert the dictionary back to a sorted list of tuples (photo_id, distance)
    results = sorted(unique_results.items(), key=lambda x: x[1])  # Sort by distance

    return results




# ----------------------------- STREAMLIT APPLICATION ----------------------------- #
# Title
st.title("🔍 Face Search Application")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    image_folder = st.text_input("Folder Path for Images", value="drive")
    generate_index = st.button("Generate Index")
    load_index = st.button("Load Existing Index")
    

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None

if 'photo_ids' not in st.session_state:
    st.session_state.photo_ids = []

if 'selected_face_embedding' not in st.session_state:
    st.session_state.selected_face_embedding = None

if 'matching_images' not in st.session_state:
    st.session_state.matching_images = []

if 'unique_faces' not in st.session_state:
    st.session_state.unique_faces = []

# Step 1: Generate or Load Index
if generate_index:
    st.session_state.index, st.session_state.photo_ids = process_and_store_images(image_folder)
elif load_index:
    index, photo_ids = load_faiss_index()
    if index is not None and photo_ids is not None:
        st.session_state.index = index
        st.session_state.photo_ids = photo_ids
    else:
        st.error("Failed to load FAISS index or photo IDs. Please check the file paths and try again.")

# Cache unique faces to avoid recomputation
@st.cache_data
def get_unique_faces(photo_ids, image_folder):
    unique_faces = []
    for photo_id in set(photo_ids):
        image_path = os.path.join(image_folder, photo_id)
        faces, boxes = detect_and_crop_faces(image_path, return_boxes=True)  # Implement this function
        for face, box in zip(faces, boxes):
            unique_faces.append((face, photo_id))
    return unique_faces

# Check if the FAISS index is available
if st.session_state.index is not None:
    # Store unique faces in session state if not already done
    if not st.session_state.unique_faces:
        st.session_state.unique_faces = get_unique_faces(st.session_state.photo_ids, image_folder)

    # Display unique faces
    st.header("📸 Unique Faces Detected")
    unique_faces = st.session_state.unique_faces

    # Display thumbnails
    cols = st.columns(10)  # 5 images per row
    for idx, (face, photo_id) in enumerate(unique_faces):
        with cols[idx % 10]:
            # Convert the face to RGB for display
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            st.image(face_rgb, caption=f"{idx}", use_column_width=True)

            # Use button to select the face for searching matches
            if st.button(f"Find", key=f"face_{idx}"):
                try:
                    # Generate embedding and search for matches
                    embedding = get_embeddings([face])[0]
                    st.session_state.selected_face_embedding = embedding
                    st.session_state.matching_images = search_similar_faces(st.session_state.index, embedding, top_k=5)
                    print(st.session_state.matching_images)
                except Exception as e:
                    st.error(f"Error processing Face {idx}: {e}")

    # Display matching images below thumbnails
    if st.session_state.matching_images:
        st.subheader("🎯 Matching Images")
        for result_path, distance in st.session_state.matching_images:
            img_path = os.path.join(image_folder, result_path)
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=f"{result_path} Distance: {distance:.4f}", use_column_width=True)
            else:
                st.warning(f"Could not load image: {result_path}")
else:
    st.warning("Please generate or load the FAISS index to proceed.")

