import dlib
import cv2

image = cv2.imread("images/321654.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()

faces = detector(gray)

for face in faces:

    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    landmarks = predictor(gray, face)
    print(landmarks.parts())
    for landmark in landmarks.parts():

        cv2.circle(image, (landmark.x, landmark.y), 2, (0, 255, 0), -1)
        print(landmark)

cv2.imshow("Image", image)
cv2.imwrite("sample.jpeg",image)
cv2.waitKey(0)
cv2.destroyAllWindows() 