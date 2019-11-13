from PIL import Image 
import cv2
import numpy as np

def get_red(img):
    redImg = img[:,:,2]
    return redImg

def get_green(img):
    greenImg = img[:,:,1]
    return greenImg

def get_blue(img):
    blueImg = img[:,:,0]
    return blueImg


imgpath = "Test_Tools/1_left.jpg"
image_file = Image.open("Test_Tools/1_left.jpg") # open colour image
img1 = image_file.convert('1') # convert image to black and white
img1.save('result.jpg')
img2 = image_file.convert('L') # convert image to black and white
img2.save('resultL.jpg')
#对图像进行二值化处理，来分割图像中的血管部分
img3=cv2.imread(imgpath)
img3=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
cv2.imwrite("result3.jpg", img3)
img4=np.array(img3)
#采用中值滤波去掉噪点
img5 = cv2.medianBlur(img4,3)
img6= cv2.medianBlur(img5,3)

#cv2.imwrite("result5.jpg", img5)
#cv2.imwrite("result6.jpg", img6)


