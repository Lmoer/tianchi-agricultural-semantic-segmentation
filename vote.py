
import os
import cv2
import torch
import random
from PIL import Image
import numpy as np
from utils.img_utils import random_scale, random_mirror_1, random_mirror_0, random_rotation, normalize, \
    generate_random_crop_pos, random_crop_pad_to_shape, random_gaussian_blur


def vote_fusion():

    data_path = '../MY/AgriculturalBrainAIChallenge/Datasets/vote'
    predict_3_1 = os.path.join(data_path, '1/image_3_predict.png') # cb 0.723
    predict_3_2 = os.path.join(data_path, '2/image_3_predict.png') # 2825 0.7277 resnet101 lovasz0.1
    predict_3_3 = os.path.join(data_path, '3/image_3_predict.png') # fusion 0.7250

    predict_4_1 = os.path.join(data_path, '1/image_4_predict.png')
    predict_4_2 = os.path.join(data_path, '2/image_4_predict.png')
    predict_4_3 = os.path.join(data_path, '3/image_4_predict.png')

    predict_3 = [predict_3_1, predict_3_2, predict_3_3]
    predict_4 = [predict_4_1, predict_4_2, predict_4_3]
    predict_ = [predict_3, predict_4]

    predict_path = '../MY/AgriculturalBrainAIChallenge/Datasets/predict/'
    predict = [os.path.join(predict_path, 'image_3_predict.png'), os.path.join(predict_path, 'image_4_predict.png')]

    for i in range(len(predict)):
        print('the', i + 1, 'image')
        result_list = []
        for j in range(len(predict_[i])):
            im = cv2.imread(predict_[i][j], 0)
            result_list.append(im)
        print('fusion number:', len(result_list))
        # each pixel
        height, width = result_list[0].shape
        vote_mask = np.zeros((height, width))
        for h in range(height):
            for w in range(width):
                record = np.zeros((1, 5))
                for n in range(len(result_list)):
                    mask = result_list[n]
                    pixel = mask[h, w]
                    # print('pix:',pixel)
                    record[0, pixel] += 1

                label = record.argmax()
                # print(label)
                vote_mask[h, w] = label

        cv2.imwrite(predict[i], vote_mask)
        print('Write', predict[i])



if __name__ == '__main__':
    vote_fusion()