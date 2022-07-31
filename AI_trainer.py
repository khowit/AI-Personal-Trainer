import cv2
import  PoseModule as pm
import numpy as np
import  time

cap = cv2.VideoCapture("image/dumbell.mp4")
detector = pm.poseDetector()

count = 0
dir = 0
pTime = 0
cTime = 0

while True:
    success, frame = cap.read()
    # frame = cv2.imread("image/arm3.jpg")
    frame = detector.findPose(frame,draw=False)
    lmList = detector.getPosition(frame,draw=False)
    if len(lmList) != 0:
        # detector.findAngle(frame,12,14,16)
        angle = detector.findAngle(frame,11,13,15)
        per = np.interp(angle,(210,280),(0,100))
        bar = np.interp(angle,(210,280),(650,100))
        # print(angle,per)
        color = (255,0,255)

        if per == 100:
            color = (0,255,0)
            if dir == 0:
                count +=0.5
                dir = 1

        if per == 0:
            color = (255,0,255)
            if dir == 1:
                count +=0.5
                dir = 0
        print(count)

        cv2.rectangle(frame, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(frame, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(frame, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

        cv2.rectangle(frame,(0, 450), (300, 720), (0,255,0),cv2.FILLED)
        if int(count) < 10:
            cv2.putText(frame, str(int(count)), (45,670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)
        else:
            cv2.putText(frame, str(int(count)), (10,670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("e"):
            break 

cap.release()
cv2.destroyAllWindows()
