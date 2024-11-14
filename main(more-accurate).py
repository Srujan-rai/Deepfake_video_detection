import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,  # Increase detection confidence
    min_tracking_confidence=0.7    # Increase tracking confidence
)

# Load real reference image and get landmarks
real_image_path = "utils/image.png"
real_image = cv2.imread(real_image_path)
real_landmarks = []

# Helper function to extract landmarks with higher resolution
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        return np.array(landmarks)
    return None

# Process the real image landmarks
real_landmarks = extract_landmarks(real_image)

# Improved similarity calculation using Euclidean Distance
def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    # Compute Euclidean distance for each landmark
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    # Calculate average of distances
    return np.mean(distances)

# Process video frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    similarities_to_real = []
    similarities_to_prev = []
    frame_rates = []
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize the frame for better resolution
        frame_resized = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2))  # Increase resolution

        landmarks = extract_landmarks(frame_resized)

        # Similarity to the real image landmarks
        sim_to_real = calculate_similarity(landmarks, real_landmarks)
        similarities_to_real.append(sim_to_real if sim_to_real is not None else 0)

        # Similarity to previous frame's landmarks
        if prev_landmarks is not None:
            sim_to_prev = calculate_similarity(landmarks, prev_landmarks)
            similarities_to_prev.append(sim_to_prev if sim_to_prev is not None else 0)
        else:
            similarities_to_prev.append(0)

        prev_landmarks = landmarks  # Update previous landmarks

        # Collect frame rate data
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_rates.append(frame_rate)

    cap.release()

    # Plot similarity scores after processing the video
    plt.figure(figsize=(10, 5))
    plt.plot(similarities_to_real, label="Similarity to Real Image", color="blue")
    plt.plot(similarities_to_prev, label="Similarity to Previous Frame", color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Scores Across Frames")
    plt.legend()
    plt.show()

    # Print average frame rate
    avg_frame_rate = np.mean(frame_rates)
    print(f"Average Frame Rate: {avg_frame_rate:.2f} fps")

# Provide the path to your video file and real image file
process_video("utils/COMBINED.mp4")
