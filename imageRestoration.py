import cv2
import numpy

#Step 1: Read Original Damaged Image
image = cv2.imread('D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\IMAGE24.png',cv2.IMREAD_UNCHANGED) 

#Step 2: Denoising using Fast Means Denoising Method
fastMeans5 = cv2.fastNlMeansDenoisingColored(image,None,5,5,7,21) 
#Chose 5 cuz smooths the image but the details are still here

#Step 3: Read Greyscaled Masked Image
#Saving denoised image:
cv2.imwrite("D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\DenoisedImage.jpg", fastMeans5)
#Reading masked images from ms Paint and making it greyscaled:
maskedImage = cv2.imread('D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\maskedImageee.png',cv2.IMREAD_GRAYSCALE) 

#Step 4: Thresholding
ret, thresh1 = cv2.threshold(maskedImage, 120, 255, cv2.THRESH_BINARY)

#Step 5: Dilation
kernel = numpy.ones((5, 5), numpy.uint8)
dilatedImage = cv2.dilate(thresh1, kernel, iterations=1)

#Step 6: Inpainting Function
#teleaImage = cv2.inpaint(maskedImage, dilatedImage, 3, cv2.INPAINT_TELEA)
teleaImage = cv2.inpaint(maskedImage, thresh1, 3, cv2.INPAINT_TELEA)
#It is found that the result without the dilation is better

#Step 7: Write Restored Image
cv2.imwrite("D:\APU Stuff\Bachelor of Computer Science (Intelligent Systems) [APD2F2202CS(IS)]\Semester 2\Imaging & Special Effects [CT029-3-2-ISE]\Group Assignment\Work\RestoredImage.jpg", teleaImage)

#Display Images (To Compare)
# concatImage = numpy.concatenate((image, fastMeans5), axis=1)
# cv2.imshow("Ori - Fast Means",concatImage)
# cv2.imshow("Masked Image", maskedImage)
# cv2.imshow("Thresholded Image", thresh1)
# cv2.imshow("Dilated Image", dilatedImage)
# cv2.imshow("Inpainted Image", teleaImage)
# cv2.waitKey()
# cv2.destroyAllWindows()

