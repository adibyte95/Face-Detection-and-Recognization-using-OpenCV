import cv2
import numpy as np 

face_cascade = cv2.CascadeClassifier('harcascade_frontalface_default.xml')
counter =0
recognizer = cv2.face.createLBPHFaceRecognizer()
#loading the training data
recognizer.load('trainingdata.yml')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cap = cv2.VideoCapture(1)
while(True):
	ret,img= cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		predict_image = np.array(gray, 'uint8')
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		id= recognizer.predict(predict_image[y: y + h, x: x + w])
		print(id)
		cv2.putText(img,str(id),(x,y+h), font,3,(255,255,255),2)
	cv2.imshow('DETECTING',img)
	if(cv2.waitKey(1) == ord('q')):
		break
	counter = counter +1
	#if counter >100:
	#	break 
cap.release()
cv2.destroyAllWindows()
