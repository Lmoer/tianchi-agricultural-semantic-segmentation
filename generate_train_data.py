import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

if __name__=='__main__':

   
    imagepath='../data/jingwei_round1_train_20190619/image_1.png'
    print('Being cut image_1.png')
    n=os.path.basename(imagepath)[:-4]
    labelname='../data/jingwei_round1_train_20190619/image_1_label.png'
    
    img = Image.open(imagepath).convert('RGB')
    mask = Image.open(labelname)
    img_1_mask = np.array(mask)
    print(img_1_mask.shape)

    img_1 = np.array(img)
    print(img_1_mask.shape,img_1.shape)
    
    for i in range(50141//512+1):
        for j in range(47161//512+1):
            img_s=img_1[i*512:min((i+1)*512,50141),j*512:min((j+1)*512,47161),:]
            img_m=img_1_mask[i*512:min((i+1)*512,50141),j*512:min((j+1)*512,47161)]
            if img_m.max()>0:
                im_name= "../data/train/data/img1"+"_"+str(i)+"_"+str(j)+".jpg"
                cv2.imwrite(im_name,img_s)
                im_name= "../data/train/label/img1"+"_"+str(i)+"_"+str(j)+".png"
                cv2.imwrite(im_name,img_m)
    
    

    imagepath='../data/jingwei_round1_train_20190619/image_2.png'
    print('Being cut image_2.png')
    n=os.path.basename(imagepath)[:-4]
    labelname='../data/jingwei_round1_train_20190619/image_2_label.png'
    
    img = Image.open(imagepath).convert('RGB')
    mask = Image.open(labelname)
    img_1_mask = np.array(mask)
    print(img_1_mask.shape)

    img_1 = np.array(img)
    print(img_1_mask.shape,img_1.shape)
    
    for i in range(40650//512+1):
        for j in range(77470//512+1):
            img_s=img_1[i*512:min((i+1)*512,40650),j*512:min((j+1)*512,77470),:]
            img_m=img_1_mask[i*512:min((i+1)*512,40650),j*512:min((j+1)*512,77470)]
            if img_m.max()>0:
                im_name= "../data/train/data/img2"+"_"+str(i)+"_"+str(j)+".jpg"
                cv2.imwrite(im_name,img_s)
                im_name= "../data/train/label/img2"+"_"+str(i)+"_"+str(j)+".png"
                cv2.imwrite(im_name,img_m)
    
    