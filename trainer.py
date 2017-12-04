# import the libraries
import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.createLBPHFaceRecognizer()
path = 'dataset'

def getImagesID(path):	
	imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
	faces = []
	IDs = []
	# names = []
	for ImagePath in imagePaths:
		
		faceImage = Image.open(ImagePath).convert('L')
		
		faceNp = np.array(faceImage,'uint8')
		ID = int(os.path.split(ImagePath)[1].split('.')[1])	
		print('Training: ',ID)	
        # print name

		faces.append(faceNp)
		IDs.append(ID)
        # names.append(name)
		cv2.imshow("Windows",faceNp)
		cv2.waitKey(10)
	return np.array(IDs), faces
	
IDs,faces =getImagesID(path)

recognizer.train(faces,IDs)
recognizer.save('trainingData.yml')
cv2.destroyAllWindows()