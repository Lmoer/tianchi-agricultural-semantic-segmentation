import numpy as np
from PIL import Image
import os
import cv2

Image.MAX_IMAGE_PIXELS = 100000000000




def empty_fill():
    print('empty_fill..............')
    data_path = 'ZhiChao_Cui/MY/AgriculturalBrainAIChallenge/Datasets/predict'
    fill_path = 'ZhiChao_Cui/MY/AgriculturalBrainAIChallenge/Datasets/predict/fill'
    predict_3 = os.path.join(data_path, 'image_3_predict.png')
    predict_4 = os.path.join(data_path, 'image_4_predict.png')
    predict_3_ = os.path.join(fill_path, 'image_3_predict.png')
    predict_4_ = os.path.join(fill_path, 'image_4_predict.png')
    predict = [predict_3, predict_4]
    predict_ = [predict_3_, predict_4_]

    for i in range(len(predict)):
        print('fill {} predict img'.format(i + 3))
        img = cv2.imread(predict[i], cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img)
        img_ = img.copy()

        shape = img_.shape
        print('shape:', shape)
        height = shape[0]
        weight = shape[1]
        #
        # length = 500
        # temp = 10
        # for x in range(0, height, length // temp):
        #     if x + length > shape[0]:
        #         break
        #     for y in range(0, weight, length // temp):
        #         if y + length > shape[1]:
        #             break
        #         x1 = img_[x:x + length, y]
        #         x1 = list(x1)
        #         x2 = img_[x:x + length, y + length]
        #         x2 = list(x2)
        #         y1 = img_[x, y:y + length]
        #         y1 = list(y1)
        #         y2 = img_[x + length, y:y + length]
        #         y2 = list(y2)
        #         if len(set(x1)) == len(set(x2)) == len(set(y1)) == len(set(y2)) == 1 and x1[0] == x2[0] == y1[0] == y2[0]:
        #             img_[x:x + length, y:y + length] = x1[0]
        #
        # length = 400
        # temp = 8
        # for x in range(0, height, length // temp):
        #     if x + length > shape[0]:
        #         break
        #     for y in range(0, weight, length // temp):
        #         if y + length > shape[1]:
        #             break
        #         x1 = img_[x:x + length, y]
        #         x1 = list(x1)
        #         x2 = img_[x:x + length, y + length]
        #         x2 = list(x2)
        #         y1 = img_[x, y:y + length]
        #         y1 = list(y1)
        #         y2 = img_[x + length, y:y + length]
        #         y2 = list(y2)
        #         if len(set(x1)) == len(set(x2)) == len(set(y1)) == len(set(y2)) == 1 and x1[0] == x2[0] == y1[0] == y2[0]:
        #             img_[x:x + length, y:y + length] = x1[0]

        length = 300
        temp = 6
        for x in range(0, height, length // temp):
            if x + length > shape[0]:
                break
            for y in range(0, weight, length // temp):
                if y + length > shape[1]:
                    break
                x1 = img_[x:x + length, y]
                x1 = list(x1)
                x2 = img_[x:x + length, y + length]
                x2 = list(x2)
                y1 = img_[x, y:y + length]
                y1 = list(y1)
                y2 = img_[x + length, y:y + length]
                y2 = list(y2)
                if len(set(x1)) == len(set(x2)) == len(set(y1)) == len(set(y2)) == 1 and x1[0] == x2[0] == y1[0] == y2[0]:
                    img_[x:x + length, y:y + length] = x1[0]

        cv2.imwrite(predict_[i], img_)
        print('write {} success!!!'.format(predict_[i]))

def labelVis():
    print('labelvis..............')
    fill_path = 'ZhiChao_Cui/MY/AgriculturalBrainAIChallenge/Datasets/predict/fill'
    predict_3_ = os.path.join(fill_path, 'image_3_predict.png')
    predict_4_ = os.path.join(fill_path, 'image_4_predict.png')
    vis_img3 = os.path.join(fill_path, 'fill_vis_3.png')
    vis_img4 = os.path.join(fill_path, 'fill_vis_4.png')
    predict_ = [predict_3_, predict_4_]
    vis_img = [vis_img3, vis_img4]

    for i in range(2):
        print('Vis', i + 3, 'image!!!')
        img = Image.open(predict_[i]).convert('L')
        img = np.asarray(img)
        print('img.shape:', img.shape)

        B = img.copy()  # blue channle
        B[B == 1] = 255
        B[B == 2] = 0
        B[B == 3] = 0
        B[B == 4] = 255
        B[B == 0] = 0

        G = img.copy()  # green channel
        G[G == 1] = 0
        G[G == 2] = 255
        G[G == 3] = 0
        G[G == 4] = 255
        G[G == 0] = 0

        R = img.copy()  # red channel
        R[R == 1] = 0
        R[R == 2] = 0
        R[R == 3] = 255
        R[R == 4] = 255
        R[R == 0] = 0

        anno_vis = np.dstack((B, G, R))
        anno_vis = cv2.resize(anno_vis, None, fx=0.03, fy=0.03)
        cv2.imwrite(vis_img[i], anno_vis)  # label image after transform
        print(vis_img[i], 'write success!!!')

def static():
    print('static..............')
    data_path = 'ZhiChao_Cui/MY/AgriculturalBrainAIChallenge/Datasets/predict'
    fill_path = 'ZhiChao_Cui/MY/AgriculturalBrainAIChallenge/Datasets/predict/fill'
    predict_3 = os.path.join(data_path, 'image_3_predict.png')
    predict_4 = os.path.join(data_path, 'image_4_predict.png')
    predict_3_ = os.path.join(fill_path, 'image_3_predict.png')
    predict_4_ = os.path.join(fill_path, 'image_4_predict.png')
    predict = [predict_3, predict_4]
    predict_ = [predict_3_, predict_4_]

    for i in range(len(predict)):
        print('static {} predict img'.format(i + 3))
        img = cv2.imread(predict[i], cv2.IMREAD_GRAYSCALE)
        img = np.asarray(img)
        img_ = cv2.imread(predict_[i], cv2.IMREAD_GRAYSCALE)
        img_ = np.asarray(img_)
        pixel0 = np.sum(img == 0)
        pixel0_ = np.sum(img_ == 0)
        pixel1 = np.sum(img == 1)
        pixel1_ = np.sum(img_ == 1)
        pixel2 = np.sum(img == 2)
        pixel2_ = np.sum(img_ == 2)
        pixel3 = np.sum(img == 3)
        pixel3_ = np.sum(img_ == 3)
        pixel4 = np.sum(img == 4)
        pixel4_ = np.sum(img_ == 4)

        print('pixel0:', pixel0 - pixel0_)
        print('pixel1:', pixel1 - pixel1_)
        print('pixel2:', pixel2 - pixel2_)
        print('pixel3:', pixel3 - pixel3_)
        print('pixel4:', pixel4 - pixel4_)


if __name__ == '__main__':
    empty_fill()
    labelVis()
    static()

















