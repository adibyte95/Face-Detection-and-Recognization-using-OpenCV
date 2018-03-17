import cv2
import numpy as np 
import pickle

# function for saving python dictionary
def save_obj(name ):
	with open("mydictionary.txt", "wb") as myFile:
		pickle.dump(name, myFile)

#function for loading dictionary
def load_obj():
    with open("mydictionary.txt", "rb") as f:
        return pickle.load(f)


eye_cascade = cv2.CascadeClassifier('harcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('harcascade_frontalface_default.xml')
id = input('enter the ID of the user \n')
name = input('enter the name of the user \n')
counter =0

# adding name to the dictionary
name_dict = load_obj()
name_dict[id] = name

save_obj(name_dict)

cap = cv2.VideoCapture(1)
while(True):
	ret,img= cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	cv2.imshow('face',img)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		roi_gray = gray[y:y+h, x:x+w]
		cv2.imwrite('dataset/user.'+id+'.'+name + '.' + str(counter)+'.jpg',roi_gray)
		counter = counter +1
	cv2.imshow('face',img)
	cv2.waitKey(10)
	#taking 100 pictures of the person
	if counter >100:
		break 
cap.release()
cv2.destroyAllWindows()
