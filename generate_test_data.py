import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

if __name__=='__main__':

   
    imagepath='../data/jingwei_round1_test_a_20190619/image_3.png'
    print('Being cut image_3.png')
    
    img = Image.open(imagepath).convert('RGB')

    img_1 = np.array(img)
    print(img_1.shape)
    
    for i in range(19903//512+1):
        for j in range(37241//512+1):
            img_s=img_1[i*512:min((i+1)*512,19903),j*512:min((j+1)*512,37241),:]
            im_name= "../data/test/image3/img3"+"_"+str(i)+"_"+str(j)+".png"
            cv2.imwrite(im_name,img_s)
            
    imagepath='../data/jingwei_round1_test_a_20190619/image_4.png'
    print('Being cut image_4.png')
    
    img = Image.open(imagepath).convert('RGB')

    img_1 = np.array(img)
    print(img_1.shape)
    
    for i in range(28832//512+1):
        for j in range(25936//512+1):
            img_s=img_1[i*512:min((i+1)*512,28832),j*512:min((j+1)*512,25936),:]
            im_name= "../data/test/image4/img4"+"_"+str(i)+"_"+str(j)+".png"
            cv2.imwrite(im_name,img_s)
    
    
    

    
    