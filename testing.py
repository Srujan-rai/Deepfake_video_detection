import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Define image and video paths
real_image_path = "utils/image.png"
video_path = "utils/COMBINED.mp4"

# Function to extract facial landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        return np.array(landmarks)
    return None

# Function to calculate similarity between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    return np.mean(distances)

# Function to detect sustained increase in variations
def detect_deepfake_start(similarities, window_size=5, threshold_multiplier=2.0):
    variations = np.abs(np.diff(similarities))
    smoothed_variations = np.convolve(variations, np.ones(window_size)/window_size, mode='valid')

    avg_variation = np.mean(smoothed_variations)
    for i in range(len(smoothed_variations)):
        if smoothed_variations[i] > threshold_multiplier * avg_variation:
            return i + window_size  # Adjust index for the window size
    return None

# Process video and generate plots
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    similarities_to_real = []
    frame_rates = cap.get(cv2.CAP_PROP_FPS)

    # Extract similarity scores for each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        sim_to_real = calculate_similarity(landmarks, real_landmarks)
        similarities_to_real.append(sim_to_real if sim_to_real is not None else 0)

    cap.release()

    deepfake_start_index = detect_deepfake_start(similarities_to_real)
    deepfake_start_time = deepfake_start_index / frame_rates if deepfake_start_index is not None else None

    variations = np.abs(np.diff(similarities_to_real))

    plt.figure(figsize=(10, 5))
    plt.plot(similarities_to_real, label="Similarity to Real Image", color="blue")
    plt.xlabel("Frame Index")
    plt.ylabel("Similarity Score")
    plt.title("Normal Similarity Plot")
    plt.legend()
    plt.savefig("normal_plot.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(similarities_to_real, label="Similarity to Real Image", color="blue")
    plt.plot(range(1, len(similarities_to_real)), variations, label="Variations", color="orange")
    if deepfake_start_index is not None:
        plt.axvline(deepfake_start_index, color="red", linestyle="--", label="Deepfake Starts")
        plt.fill_between(
            range(deepfake_start_index, len(similarities_to_real)),
            0, max(similarities_to_real),
            color="yellow", alpha=0.3, label="Deepfake Region"
        )
    plt.xlabel("Frame Index")
    plt.ylabel("Scores / Variations")
    plt.title("Analysis Plot with Detected Deepfake Region")
    plt.legend()
    plt.savefig("analysis_plot.png")
    plt.show()

    if deepfake_start_time is not None:
        print(f"Deepfake starts at frame index: {deepfake_start_index}")
        print(f"Deepfake starts approximately at: {deepfake_start_time:.2f} seconds")
    else:
        print("No significant sustained variation detected to indicate deepfake start.")

real_image = cv2.imread(real_image_path)
real_landmarks = extract_landmarks(real_image)

# Run the video processing
process_video(video_path)
