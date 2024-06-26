import cv2
import mediapipe as mp
import time


class FacemeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionCon=minDetectionCon
        self.minTrackCon=minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.minDetectionCon,self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self,image,draw=False):
        self.imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(self.imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(image, faceLms, self.mpFaceMesh.FACE_CONNECTIONS(),
                                      self.drawSpec, self.drawSpec)

                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = image.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    print(id, x, y)










def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    while True:
        success, image = cap.read()
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 3)
        cv2.imshow("image", image)
        cv2.waitKey(1)

if __name__== "__main__":
    main()