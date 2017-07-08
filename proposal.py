# -*- coding: utf-8 -*-
from glob import glob
import os
import numpy as np
from PIL import Image


def generate_proposal(data_path, save_path, resize_ratio_threshold=1.5, resize_h_w_ratio=0.8):
    """
    generate proposals for test images. proposal images will saved in save_path.
    If test image is too large\narrow\fat, several proposals are generated.
    :param data_path:               path of data to be processed. str.
    :param save_path:               path of data to be saved. str.
    :param resize_ratio_threshold:  threshold of large image or not, >1.0. float. default:1.5
    :param resize_h_w_ratio:        threshold of narrow\fat image or not, a height/width ratio
                                    regarding to train images. float. default:0.6
                                    e.g.: the height/width ratio of train images is 112/92.
                                    Image with aspect ratio >112/92*0.6 will be treated as fat,
                                    Image with aspect ratio <112/92*1.4 will be treated as slim,
    :return:                        None
    """
    HEIGHT = 112
    WIDTH = 92
    o_height = HEIGHT * resize_ratio_threshold
    o_width = WIDTH * resize_ratio_threshold
    o_h_w_ratio = 1.0 * HEIGHT / WIDTH
    img_files = glob(os.path.join(data_path, "*.jpg"))
    proposal_shape = np.array([(0.25, 0.25, 0.75, 0.75), # middle
                      (0, 0, 0.5, 0.5),         # upper left
                      (0.25, 0, 0.75, 0.5),       # upper middle
                      (0.5, 0, 1, 0.5),         # upper right
                      (0, 0.5, 0.5, 1),         # bottom left
                      (0.5, 0.5, 1, 1)          # bottom right
                      ])
    proposal_shape_fat = np.array([(0, 0, 0.4, 1),                # left
                             (0.3, 0, 0.7, 1),          # middle
                             (0.7, 0, 1, 1),            # right
                      ])
    proposal_shape_slim = np.array([(0, 0, 1, .4),               # top
                             (0, .4, 1, .7),            # middle
                             (0, 0.7, 1, 1),            # bottom
                      ])
    for img in img_files:
        img = Image.open(img)
        name = img.filename.split('\\')[1]
        # name_id = name.split('.')[0]
        name_id = name.split('_')[0]
        if img.mode not in ('L', 'RGB'):
            img = img.convert('RGB')
        # 判断长宽比, 长宽比接近训练集
        if (img.height*1.0/img.width > o_h_w_ratio*resize_h_w_ratio and
            img.height * 1.0 / img.width < o_h_w_ratio * (2 - resize_h_w_ratio)):
            # 判断大小，大小接近训练集，不用proposal
            if img.height < o_height and img.width < o_width:
                img2 = img
                path = os.path.join(save_path, name_id + 'p0.jpg')
                img2.save(path, format='JPEG')
            # 大小不接近训练集, 原图resize
            if not(img.height < o_height and img.width < o_width):
                img2 = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
                path = os.path.join(save_path, name_id + 'p0.jpg')
                img2.save(path, format='JPEG')
                for i in range(len(proposal_shape)):
                    p = img.crop(np.array([img.width, img.height, img.width, img.height])* proposal_shape[i])
                    path = os.path.join(save_path, name_id + 'p'+str(i+1)+'.jpg')
                    p.save(path, format='JPEG')
        # 长宽比不接近训练集, fat
        elif img.height * 1.0 / img.width < o_h_w_ratio * resize_h_w_ratio:
            # 原图resize
            img2 = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            path = os.path.join(save_path, name_id + 'p0.jpg')
            img2.save(path, format='JPEG')
            for i in range(len(proposal_shape_fat)):
                p = img.crop(np.array([img.width, img.height, img.width, img.height]) * proposal_shape_fat[i])
                path = os.path.join(save_path, name_id + 'p' + str(i+1) + '.jpg')
                p.save(path, format='JPEG')
        # slim
        else:
            # 原图resize
            img2 = img.resize((HEIGHT, WIDTH), Image.ANTIALIAS)
            path = os.path.join(save_path, name_id + 'p0.jpg')
            img2.save(path, format='JPEG')
            for i in range(len(proposal_shape_slim)):
                p = img.crop(np.array([img.width, img.height, img.width, img.height]) * proposal_shape_slim[i])
                path = os.path.join(save_path, name_id + 'p' + str(i+1) + '.jpg')
                p.save(path, format='JPEG')


# generate_proposal('./test0/', save_path='./test/', resize_h_w_ratio=0.6)