from urllib.parse import _ResultMixinStr
import numpy as np
import cv2
from cv2 import dnn

#import in the model file paths
proto_file = 'D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\colorization_deploy_v2.prototxt'
model_file = 'D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\colorization_release_v2.caffemodel'
hull_pts = 'D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\pts_in_hull.npy'
img_path = 'D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\RestoredImage.jpg'

#Read the model parameters
net = dnn.readNetFromCaffe(proto_file,model_file)
kernel = np.load(hull_pts)

#Reading and preprocessing image
img = cv2.imread(img_path)
scaled = img.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

#Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

#Resizing images
resized = cv2.resize(lab_img, (224, 224))

#Split the L channel
L = cv2.split(resized)[0]

#Mean subtraction
L -= 50

#Predicting the ab channels from the input L channel
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

#Resize the predicted 'ab' volume to the same dimensions as the input image
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

#Take the L channel from the image
L = cv2.split(lab_img)[0]\
    
#Join the L channel with predicted ab channel
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

#Convert the image from Lab to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

#Change the image to 0-255 range and convert it from float32 to int
colorized = (255 * colorized).astype("uint8")

#Print the Image Out
cv2.imwrite("D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\ColouredImage.jpg", colorized)
