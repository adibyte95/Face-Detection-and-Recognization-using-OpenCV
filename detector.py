import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('harcascade_frontalface_default.xml')
counter =0
recognizer = cv2.face.LBPHFaceRecognizer_create()
#loading the training data
recognizer.read('trainingdata.yml')
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


#function for loading dictionary
def load_obj():
    with open("mydictionary.txt", "rb") as fileName:
        return pickle.load(fileName)


#opening the webcam for the image feed
cap = cv2.VideoCapture(1)
while(True):
	ret,img= cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		predict_image = np.array(gray, 'uint8')
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		#predicting the id of the corresponding face
		id= recognizer.predict(predict_image[y: y + h, x: x + w])
		name_dict = load_obj()
		# getting the name of the person
		name = name_dict[str(id[0])]
		print('name of the person detected : ',name)
		#showing the name of the person in the OpenCV window
		cv2.putText(img,str(name),(x,y+h), font,3,(255,255,255),2)
	cv2.imshow('DETECTING',img)
	if(cv2.waitKey(1) == ord('q')):
		break
	counter = counter +1
	#if counter >100:
	#	break 
cap.release()
cv2.destroyAllWindows()
