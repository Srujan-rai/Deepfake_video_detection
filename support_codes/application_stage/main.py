import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")  # Load the 68-point model

# Load the real reference image and compute its landmarks
real_image_path = "utils/image.png"
real_image = cv2.imread(real_image_path)
gray_real = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
faces_real = detector(gray_real)
real_landmarks = []

if faces_real:
    shape = predictor(gray_real, faces_real[0])
    real_landmarks = np.array([(p.x, p.y) for p in shape.parts()])

def calculate_similarity(landmarks1, landmarks2):
    # Calculate Euclidean distances between corresponding points
    if len(landmarks1) == 0 or len(landmarks2) == 0:
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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        if len(faces) > 0:
            shape = predictor(gray, faces[0])
            landmarks = np.array([(p.x, p.y) for p in shape.parts()])
            
            # Similarity to the real image landmarks
            sim_to_real = calculate_similarity(landmarks, real_landmarks)
            similarities_to_real.append(sim_to_real if sim_to_real is not None else 0)
            
            # Similarity to previous frame's landmarks
            if prev_landmarks is not None:
                sim_to_prev = calculate_similarity(landmarks, prev_landmarks)
                similarities_to_prev.append(sim_to_prev if sim_to_prev is not None else 0)
            else:
                similarities_to_prev.append(0)  # Initial frame with no previous comparison
            
            prev_landmarks = landmarks  # Update the previous landmarks
            
        else:
            similarities_to_real.append(0)
            similarities_to_prev.append(0)

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
process_video("utils/fake.mp4")
