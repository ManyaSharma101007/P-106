import cv2

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cap = cv2.VideoCapture('walking.avi')

while True:
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGRGRAY)

    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    
    for (x,y,w,h) in bodies:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)  
       roi_color = frame[y:y+h, x:x+w]
    
    cv2.imshow('img',frame)

    if cv2.waitKey(1) == 32: 
        break

cap.release()
cv2.destroyAllWindows()
