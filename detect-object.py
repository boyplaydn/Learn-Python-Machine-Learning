import cv2
import numpy as np


kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    flip_img = cv2.flip(img, 1)
    hsv = cv2.cvtColor(flip_img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([10,50,50])
    upper_red = np.array([90,255,255])
    lowerMask = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([255,255,255])
    upperMask = cv2.inRange(hsv, lower_red, upper_red)

    mask = lowerMask + upperMask
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_OPEN, kernelClose)
    maskFinal = maskClose

    ctrs,_ = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    i = 0
    for rect in rects:
        i = i + 1
        print(rect[0],'-',rect[1])
        print(rect[0]+rect[2] ,'-', rect[1] + rect[3])
        cv2.rectangle(flip_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 10)
        cv2.putText(flip_img, str(i+1),(rect[0],rect[1] + rect[3]), font, 1,(0,0,0),2,cv2.LINE_AA)

    cv2.imshow('image',flip_img)
    cv2.imshow('open',maskOpen)
    cv2.imshow('close',maskClose)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()