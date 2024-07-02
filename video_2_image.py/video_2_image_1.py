import cv2
import os

output_dir = 'video_2_image/output'


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture('videos/WhatsApp Video 2024-07-02 at 7.05.17 PM.mp4')
count = 0

while True:
    success, image = cap.read()
    
    if not success:
        break  

    cv2.imwrite(os.path.join(output_dir, f"frame{count}.jpg"), image)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
