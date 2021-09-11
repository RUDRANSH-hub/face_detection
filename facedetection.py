import os

import cv2
import numpy as np

alg="haarcascade_frontalface_default.xml" 
haar_cascade=cv2.CascadeClassifier(alg)  
dataset="datasets"  #making a folder in which our train data is captured
sub_data="sunney leone"
path=os.path.join(dataset,sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
(width,height)=(130,100)
  
cam=cv2.VideoCapture(0)
# (image,label,name,id)=([],[],{},0)  #making emptyy folders
# for (subdir,dir,file) in os.walk(dataset):    #it iterate over datasets and assigh names
#     for i in dir:
#         name[id]=i
#         new_path=os.path.join(dataset,i) #join attach the 2 path and make a new path dir or pwd
#         for x in os.listdir(new_path):
#             new_new_path=new_path + '/' + str(x)  #iterating over images by giving path
#             N=id #id initilize to n and n use as int in label
#             image.append(cv2.imread(new_new_path,0))
#             label.append(int(N))
#         id+=1
# (image,label)=[np.array(arr) for arr in [image,label]]
# print(image,label)
# model=cv2.face.FisherFaceRecognizer_create()
# model.train(image,label)


count=0
while count<51:
    print(count)
    (_,img)=cam.read()
    
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=haar_cascade.detectMultiScale(grayimg,1.3,4)
    for x,y,w,h in face:

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        slice_face=grayimg[y:y+h ,x:x+w]
        face_resize=cv2.resize(slice_face,(width,height))
        cv2.imwrite("%s/%s.png" % (path,count),face_resize)
        count+=1
        # print(model.pridict(face_resize))
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

        
    cv2.imshow("face_detection",img)


    key=cv2.waitKey(10)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
