from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure uploads directory exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mediapipe Face Mesh Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Extract landmarks from an image
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        return np.array(landmarks)
    return None

# Calculate similarity between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    return np.mean(distances)

# Process video and calculate similarity scores
def process_video(video_path, real_landmarks):
    cap = cv2.VideoCapture(video_path)
    similarities_to_real = []
    similarities_to_prev = []
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for better resolution
        frame_resized = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))

        # Extract landmarks
        landmarks = extract_landmarks(frame_resized)

        # Calculate similarity to real image
        sim_to_real = calculate_similarity(landmarks, real_landmarks)
        similarities_to_real.append(sim_to_real if sim_to_real is not None else 0)

        # Calculate similarity to previous frame
        if prev_landmarks is not None:
            sim_to_prev = calculate_similarity(landmarks, prev_landmarks)
            similarities_to_prev.append(sim_to_prev if sim_to_prev is not None else 0)
        else:
            similarities_to_prev.append(0)

        prev_landmarks = landmarks

    cap.release()

    return similarities_to_real, similarities_to_prev

# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")

# Route to handle file uploads and processing
@app.route("/upload", methods=["POST"])
def upload():
    real_image = request.files.get("real_image")
    video_file = request.files.get("video_file")

    if not real_image or not video_file:
        return jsonify({"error": "Both files are required"}), 400

    # Save uploaded files
    real_image_path = os.path.join(UPLOAD_FOLDER, "real_image.png")
    video_file_path = os.path.join(UPLOAD_FOLDER, "uploaded_video.mp4")
    real_image.save(real_image_path)
    video_file.save(video_file_path)

    # Load and process real image
    real_image = cv2.imread(real_image_path)
    real_landmarks = extract_landmarks(real_image)
    if real_landmarks is None:
        return jsonify({"error": "No face detected in the real image"}), 400

    # Process the video
    similarities_to_real, similarities_to_prev = process_video(video_file_path, real_landmarks)

    # Return results as JSON
    return jsonify({
        "similarities_to_real": similarities_to_real,
        "similarities_to_prev": similarities_to_prev
    })

if __name__ == "__main__":
    app.run(debug=True)
