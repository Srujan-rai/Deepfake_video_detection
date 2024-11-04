import cv2
import os
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize MediaPipe face mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Paths for input videos
real_video_path = 'real.mp4'  # Path to your real video
fake_video_path = 'fake.mp4'  # Path to your fake video
output_folder_real = 'video_2_image/real_frames'
output_folder_fake = 'video_2_image/fake_frames'
detected_faces_folder_real = 'face_recognition/real_faces'
detected_faces_folder_fake = 'face_recognition/fake_faces'

# Create directories if they don't exist
os.makedirs(output_folder_real, exist_ok=True)
os.makedirs(output_folder_fake, exist_ok=True)
os.makedirs(detected_faces_folder_real, exist_ok=True)
os.makedirs(detected_faces_folder_fake, exist_ok=True)

# Step 1: Convert Videos to Frames
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}.")

# Step 2: Detect Faces and Save Detected Faces
def detect_faces_and_save(input_folder, output_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, img)
    
    print(f"Face detection complete in {input_folder}. Frames with detected faces are saved.")

# Step 3: Extract Face Mesh Data
def extract_face_mesh_data(input_folder):
    face_mesh_data = {}
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(input_folder, filename)
            image = cv2.imread(img_path)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = faceMesh.process(imgRGB)
            if results.multi_face_landmarks:
                landmarks = []
                for faceLms in results.multi_face_landmarks:
                    for lm in faceLms.landmark:
                        ih, iw, _ = image.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        landmarks.append((x, y))
                
                face_mesh_data[filename] = landmarks

    print("Face mesh data extraction complete.")
    return face_mesh_data

# Step 4: Calculate Similarity between Real and Fake Face Mesh Data
def calculate_similarity(real_face_mesh, test_face_mesh):
    if len(real_face_mesh) != len(test_face_mesh):
        raise ValueError("Landmark counts do not match.")
    
    total_distance = 0
    for real_landmark, test_landmark in zip(real_face_mesh, test_face_mesh):
        distance = np.linalg.norm(np.array(real_landmark) - np.array(test_landmark))
        total_distance += distance
    
    average_distance = total_distance / len(real_face_mesh)
    similarity_score = 1 / (1 + average_distance)  # Normalize score between 0 and 1
    return similarity_score

# Plotting function
def plot_similarity_scores(scores):
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(10, 6))
    
    for i, score in enumerate(scores):
        plt.clf()  # Clear the figure
        plt.plot(range(len(scores)), scores, marker='o', linestyle='-', color='b')
        plt.xlim(0, len(scores))
        plt.ylim(0, 1)
        plt.title("Similarity Score Over Frames")
        plt.xlabel("Frame Number")
        plt.ylabel("Similarity Score")
        plt.axhline(y=0.5, color='r', linestyle='--')  # Threshold line
        plt.pause(0.1)  # Pause to update the plot
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Main function to run the complete process
def main():
    # Step 1: Extract frames from real and fake videos
    extract_frames(real_video_path, output_folder_real)
    extract_frames(fake_video_path, output_folder_fake)

    # Step 2: Detect faces in extracted frames
    detect_faces_and_save(output_folder_real, detected_faces_folder_real)
    detect_faces_and_save(output_folder_fake, detected_faces_folder_fake)

    # Step 3: Extract face mesh data from detected faces
    face_mesh_data_real = extract_face_mesh_data(detected_faces_folder_real)
    face_mesh_data_fake = extract_face_mesh_data(detected_faces_folder_fake)

    # Step 4: Compare detected face mesh data for corresponding frames
    similarity_scores = []
    frame_count = min(len(face_mesh_data_real), len(face_mesh_data_fake))  # Compare only up to the shorter video length

    for frame_num in range(frame_count):
        real_frame_name = f"frame_{frame_num:04d}.jpg"
        test_frame_name = f"frame_{frame_num:04d}.jpg"

        if real_frame_name in face_mesh_data_real and test_frame_name in face_mesh_data_fake:
            real_face_mesh = face_mesh_data_real[real_frame_name]
            test_face_mesh = face_mesh_data_fake[test_frame_name]
            similarity_score = calculate_similarity(real_face_mesh, test_face_mesh)
            similarity_scores.append(similarity_score)
            print(f"Similarity score for frame {frame_num}: {similarity_score:.2f}")

            # Decide if the video frame is real or fake based on the score
            if similarity_score < 0.5:  # Example threshold
                print(f"The frame {frame_num} is likely FAKE.")
            else:
                print(f"The frame {frame_num} is likely REAL.")

    # Step 5: Plot the similarity scores
    plot_similarity_scores(similarity_scores)

if __name__ == "__main__":
    main()
