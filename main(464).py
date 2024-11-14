import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


real_image_path = "utils/image.png"
real_image = cv2.imread(real_image_path)
real_landmarks = []


def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = [(lm.x, lm.y) for lm in results.multi_face_landmarks[0].landmark]
        return np.array(landmarks)
    return None


real_landmarks = extract_landmarks(real_image)


def calculate_similarity(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    return np.mean(distances)


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

        landmarks = extract_landmarks(frame)

        sim_to_real = calculate_similarity(landmarks, real_landmarks)
        similarities_to_real.append(sim_to_real if sim_to_real is not None else 0)

        if prev_landmarks is not None:
            sim_to_prev = calculate_similarity(landmarks, prev_landmarks)
            similarities_to_prev.append(sim_to_prev if sim_to_prev is not None else 0)
        else:
            similarities_to_prev.append(0)

        prev_landmarks = landmarks  # Update previous landmarks

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_rates.append(frame_rate)

    cap.release()

    plt.figure(figsize=(10, 5))
    plt.plot(similarities_to_real, label="Similarity to Real Image", color="blue")
    plt.plot(similarities_to_prev, label="Similarity to Previous Frame", color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Similarity Score")
    plt.title("Similarity Scores Across Frames")
    plt.legend()
    plt.show()

    avg_frame_rate = np.mean(frame_rates)
    print(f"Average Frame Rate: {avg_frame_rate:.2f} fps")

process_video("COMBINED.mp4")
