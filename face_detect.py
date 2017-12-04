import cv2
import numpy as np 

eye_cascade = cv2.CascadeClassifier('harcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('harcascade_frontalface_default.xml')
id = input('enter the ID of the user')
counter =0

cap = cv2.VideoCapture(1)
while(True):
	ret,img= cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	cv2.imshow('face',img)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		roi_gray = gray[y:y+h, x:x+w]
		cv2.imwrite('dataset/user.'+id+'.'+str(counter)+'.jpg',roi_gray)
		counter = counter +1
	cv2.imshow('face',img)
	cv2.waitKey(100)
	if counter >100:
		break 
cap.release()
cv2.destroyAllWindows()
