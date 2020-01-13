import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('1533630293571823615.mp4')
i = 0
t = 0
done = False
#cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()

    flip_frame = cv2.flip(frame, 1)
    blur = cv2.GaussianBlur(flip_frame, (11, 11), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array([10, 120, 30])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    ball = cv2.bitwise_and(flip_frame, flip_frame, mask=mask)
    ball_crts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_crts = imutils.grab_contours(ball_crts)

    font = cv2.FONT_HERSHEY_COMPLEX
    if(len(ball_crts) > 0):
        c = max(ball_crts, key=cv2.contourArea)
        ((x,y), radius)=cv2.minEnclosingCircle(c)
        M =cv2.moments(c)
        ball_center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
        if(radius>10):
            cv2.circle(flip_frame, (int(x), int(y)), int(radius), (255,0,0), 2)
            cv2.circle(flip_frame, ball_center, 5, (255,0,0), -1)


    if int(M["m10"]/M["m00"]) >= 400 and int(M["m10"]/M["m00"]) <= 430 :
        done = True
    if int(M["m10"]/M["m00"]) > 430 and done:
        t = t + 1
        done = False
    if int(M["m10"]/M["m00"]) < 400 and done:
        i= i + 1
        done = False
           
    cv2.putText(flip_frame, str(i),(130,100), font, 1,(0,0,0),2,cv2.LINE_AA) 
    cv2.putText(flip_frame, str(t),(730,100), font, 1,(0,0,0),2,cv2.LINE_AA)
    cv2.line(flip_frame, (430, 2), (400, 473), (0,255,0), 2)
    
    

    cv2.imshow('ball',ball)
    cv2.imshow('mask',mask)
    cv2.imshow('flip_frame',flip_frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()