
from __future__ import division,print_function
import os
import numpy as np
from PIL import Image, ImageFilter
import cv2

img_path = 'C:/Users/upstar/Desktop/tensorflow_models_learning-master/dataset/train/0/'

def convert(fname, crop_size):
    img = Image.open(fname)
    debug = 1
    blurred = img.filter(ImageFilter.BLUR) 

    ba = np.array(blurred)
    #ba = np.array(img)

    h, w, _ = ba.shape
    if debug>0:
        print("h=%d, w=%d"%(h,w))
    #这里的1.2, 32, 5, 0.8都是后续可以调整的参数。 只是暂时觉得用这个来提取背景不错。
    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)
        foreground = (ba > max_bg + 5).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()
 
        if debug>0:
            print(foreground, left_max, right_max, bbox)
        if bbox is None:
            print('bbox none for {} (???)'.format(fname))
        else:
            left, upper, right, lower = bbox
            #如果弄到的框小于原图的80%，很可能出bug了，就舍弃这个框。
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print('bbox too small for {}'.format(fname))
                bbox = None
    else:
        bbox = None
 
    if bbox is None:
        if debug>0:
            print 
        bbox = square_bbox(img)
 
    cropped = img.crop(bbox)
    resized = cropped.resize([crop_size, crop_size])
    return resized
 
def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)

def main():
#C:\Users\upstar\Pictures\Eye_Aging\Dataset_Test
#cwd='/test/'  
#人为设定2类

    for img_name in os.listdir(img_path):
        img_file=img_path+'/'+img_name #每一个图片的地址   
        img = convert(img_file,299)
        '''逆时针旋转90度的新Image图像'''
        #img = Image.open(img_file)
        #img.rotate(90).save('C:/Users/upstar/OneDrive/pic/Convert90/'+img_name)
        '''逆时针旋转180度的新Image图像'''
        #img.rotate(180).save('C:/Users/upstar/OneDrive/pic/Convert180/'+img_name)
        '''逆时针旋转270度的新Image图像'''
        #img.rotate(270).save('C:/Users/upstar/OneDrive/pic/Convert270/'+img_name)
        
#对图像进行二值化处理，来分割图像中的血管部分
        img = cv2.imread(img_file)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        xImg = cv2.flip(img,1,dst=None) #水平镜像
        xImg1 = cv2.flip(img,0,dst=None) #垂直镜像
        cv2.imwrite(img_path+'/1_'+img_name, xImg)
        cv2.imwrite(img_path+'/2_'+img_name, xImg1)
        print(np.shape(img))
        print(img_path+img_name)

 
if __name__=='__main__':
    
    main()
