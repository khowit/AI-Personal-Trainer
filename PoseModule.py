import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5 , trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, frame, draw=True):
    
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return frame

    def getPosition(self, frame,draw=True):
    
        self.lmList = []
        if self.results.pose_landmarks:
            myHand = self.results.pose_landmarks
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w),int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)

        return self.lmList

    def findAngle(self, frame, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw
        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(frame, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def angleCheck(self, myAngle, targetAngle, addOn=20):
        return targetAngle - addOn < myAngle < targetAngle + addOn


def main():
    cap = cv2.VideoCapture('image/danc.mp4')
    pTime = 0
    cTime = 0
    detector = poseDetector()

    while True:
        success, frame = cap.read()
        frame = detector.findPose(frame)
        lmList = detector.getPosition(frame)
        if len(lmList) != 0:
            cv2.circle(frame,(lmList[14][1], lmList[14][2]),15,(0,0,255),cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Output", frame)
        if cv2.waitKey(10) & 0xFF == ord("e"):
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()