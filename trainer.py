# import the libraries
import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def getImagesID(path):	
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []
	names = []
	for ImagePath in imagePaths:
		
		faceImage = Image.open(ImagePath).convert('L')
		
		faceNp = np.array(faceImage,'uint8')
		
		# id
		ID = int(os.path.split(ImagePath)[1].split('.')[1])	
		print('Training: ',ID)	
       
	    #  name
		name = os.path.split(ImagePath)[1].split('.')[2]
		names.append(name)

		faces.append(faceNp)
		IDs.append(ID)
        # names.append(name)
		cv2.imshow("Windows",faceNp)
		cv2.waitKey(10)
	return np.array(IDs), np.array(names), faces
	
IDs,names, faces =getImagesID(path)
print(names)

#training the recognizer
recognizer.train(faces,IDs)
recognizer.write('trainingData.yml')
cv2.destroyAllWindows()