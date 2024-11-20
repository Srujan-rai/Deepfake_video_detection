from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import mediapipe as mp
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure uploads and processed directory exists
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

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
        return np.array(landmarks), results.multi_face_landmarks[0]
    return None, None

# Calculate similarity between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    return np.mean(distances)

# Process video, calculate similarity scores, and save face mesh overlay video
def process_video(video_path, real_landmarks):
    import time  # To measure processing time

    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    # Retrieve video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Verify valid video properties
    if frame_width <= 0 or frame_height <= 0 or fps <= 0:
        raise ValueError("Invalid video properties. Ensure the input video is valid.")

    # Output video setup
    output_path = os.path.join("processed", "processed_video.mp4")
    os.makedirs("processed", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    similarities_to_real = []
    similarities_to_prev = []
    prev_landmarks = None

    # Debugging variables
    start_time = time.time()
    frame_idx = 0

    # Processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Debug: Display progress
        frame_idx += 1
        print(f"Processing frame {frame_idx}/{frame_count}...")

        # Extract landmarks
        landmarks, face_mesh_landmarks = extract_landmarks(frame)

        # Draw face mesh (debugging visualization)
        if face_mesh_landmarks:
            for landmark in face_mesh_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Debug: Verify processed frame integrity
        cv2.imshow("Processed Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit during debugging
            break

        # Write frame to the output video
        out.write(frame)

        # Calculate similarity
        sim_to_real = calculate_similarity(landmarks, real_landmarks)
        similarities_to_real.append(sim_to_real if sim_to_real is not None else 0)

        if prev_landmarks is not None:
            sim_to_prev = calculate_similarity(landmarks, prev_landmarks)
            similarities_to_prev.append(sim_to_prev if sim_to_prev is not None else 0)
        else:
            similarities_to_prev.append(0)

        prev_landmarks = landmarks

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Debug: Measure processing time
    total_time = time.time() - start_time
    print(f"Processed {frame_idx} frames in {total_time:.2f} seconds.")

    # Verify output file
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Processed video was not saved correctly at: {output_path}")

    print(f"Processed video saved at: {output_path}")

    return similarities_to_real, similarities_to_prev, output_path


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
    real_landmarks, _ = extract_landmarks(real_image)
    if real_landmarks is None:
        return jsonify({"error": "No face detected in the real image"}), 400

    # Process the video
    similarities_to_real, similarities_to_prev, processed_video_path = process_video(video_file_path, real_landmarks)

    # Return results as JSON
    return jsonify({
        "similarities_to_real": similarities_to_real,
        "similarities_to_prev": similarities_to_prev,
        "processed_video_url": processed_video_path
    })

if __name__ == "__main__":
    app.run(debug=True)
