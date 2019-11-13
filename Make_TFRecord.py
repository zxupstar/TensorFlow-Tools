import os 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np

def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values): 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def image_to_tfexample(image_data, label,size):    
    return tf.train.Example(features=tf.train.Features(feature={        
    'image': bytes_feature(image_data),        
    'label': int64_feature(label),        
    'image_width':int64_feature(size[0]),        
    'image_height':int64_feature(size[1])
    }))


#cwd='/test/'
cwd = 'C:/Users/upstar/OneDrive/VScode/Eye_Aging/Heath_2Row/Dataset_Test'

classes={'0','1'}  #人为设定2类
#writer= tf.python_io.TFRecordWriter("*.tfrecords") #要生成的文件
writer= tf.python_io.TFRecordWriter("C:/Users/upstar/OneDrive/VScode/Eye_Aging/Heath_2Row/Health_Unhealth_Test.tfrecords") #要生成的文件

for index,name in enumerate(classes):
    class_path=cwd+name+'/'
    for img_name in os.listdir(class_path): 
        img_path=class_path+img_name #每一个图片的地址

        img=Image.open(img_path)
        img= img.resize((128,128))
        print(np.shape(img))
        img_raw=img.tobytes()#将图片转化为二进制格式
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        })) #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串

writer.close()