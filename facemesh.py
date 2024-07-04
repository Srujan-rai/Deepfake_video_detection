import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)
pTime=0
mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=1)

while True:
    success,image=cap.read()
    imgRGB=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image,faceLms,mpFaceMesh.FACEMESH_CONTOURS ,
                                  drawSpec,drawSpec)

            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                ih,iw,ic=image.shape
                x,y=int(lm  .x*iw),int(lm.y*ih)
                print(id,x,y)












    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(image,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)
    cv2.imshow("image",image)
    cv2.waitKey(1)