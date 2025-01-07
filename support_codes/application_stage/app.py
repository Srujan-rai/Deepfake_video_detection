from flask import Flask, request, jsonify
import cv2
import os
import threading

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed_faces'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Shared variable for process status
process_status = []

@app.route('/upload', methods=['POST'])
def upload_files():
    global process_status
    process_status = []  # Reset process log
    
    if 'video' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'Video and image files are required'}), 400
    
    video_file = request.files['video']
    image_file = request.files['image']
    
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    
    video_file.save(video_path)
    image_file.save(image_path)
    
    # Start processing in a separate thread
    threading.Thread(target=process_video, args=(video_path,)).start()
    
    return jsonify({'message': 'Files uploaded successfully'}), 200

def process_video(video_path):
    global process_status
    process_status.append("Video processing started.")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    face_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        process_status.append(f"Processing frame {frame_count}")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_count += 1
                face_path = os.path.join(PROCESSED_FOLDER, f"face_{frame_count}_{face_count}.jpg")
                cv2.imwrite(face_path, face_img)
                process_status.append(f"Face detected in frame {frame_count}. Saved face_{frame_count}_{face_count}.jpg")
    
    cap.release()
    process_status.append("Video processing completed.")

@app.route('/process_status', methods=['GET'])
def get_process_status():
    return jsonify({"status": process_status})

if __name__ == '__main__':
    app.run(debug=True)
    