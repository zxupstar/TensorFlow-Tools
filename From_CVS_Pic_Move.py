import csv
import shutil
import os



target_path = 'C:/Users/upstar/Desktop/tensorflow_models_learning-master/dataset/all/'
#strings = ['512c','90c','180c','270c']
#C:\Users\upstar\Desktop\tensorflow_models_learning-master\dataset\all
original_path = 'C:/Users/upstar/Desktop/tensorflow_models_learning-master/dataset/all/'
with open('C:/Users/upstar/Desktop/tensorflow_models_learning-master/dataset/all/Sex_2Row.csv',"rt", encoding="utf-8-sig") as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    for row in rows:
        #for i in strings :
        if os.path.exists(target_path+row[1] ) == False :
            os.makedirs(target_path+row[1])
        target_floder = target_path+row[1]+'/'
        original_file = original_path +'90c'+ row[0]
        print(original_file)
        try:
            shutil.move(original_file,target_floder)
        except Exception :
            print("NOT Found")
       
