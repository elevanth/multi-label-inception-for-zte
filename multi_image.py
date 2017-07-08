# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import scipy.io as sio
# import colorsys
# import cv2
import os
import sys

MODE = "test"
folder_path = "test" # sys.argv[1]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("labels.txt")]

files = os.listdir(folder_path)
length = len(files)
batch_size = 128
folds_num = int(length / batch_size) + 1

test_mat_path = 'tags.mat'
mat_data = sio.loadmat(test_mat_path)
test_data = mat_data['beta_tags_cell']
test_num = test_data.shape[1]
test_ground_truth = {}
for num in range(test_num):
    photo_name = test_data[0][num][0]
    man_label = test_data[1][num][0][0]
    hat_label = test_data[2][num][0][0]
    glass_label = test_data[3][num][0][0]
    mask_label = test_data[4][num][0][0]
    tmglass_label = test_data[5][num][0][0]
    test_labels = [man_label, hat_label, glass_label, mask_label, tmglass_label]
    test_ground_truth[photo_name] = test_labels


TF_dict={True:1, False:0}
image_tensor = {}

def yon(YN):
    if YN:
        return "1"
    else:
        return "0"

def getTensor(num):
    img_tensor = {}
    batch_list = files[num*batch_size : min((num+1)*batch_size, length)]
    for image_path in batch_list:
        real_image_path = folder_path + "/" + image_path
        image_data = tf.gfile.FastGFile(real_image_path, 'rb').read()
        img_tensor[image_path] = image_data
    return img_tensor

# for image_path in os.listdir(folder_path):
#     real_image_path = folder_path + "\\" + image_path
#     # os.system('python single_image.py %s' %(real_image_path))
#     image_data = tf.gfile.FastGFile(real_image_path, 'rb').read()
#     image_tensor[image_path] = image_data

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    diff = np.zeros([1, 4])
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    # filename = "results.txt"
    # with open(filename, 'a+') as f:
    #     f.write('\n%-13s%-9s%-7s%-7s%-7s' \
    #         % ("FILES", "GENDER", "GLASS", "MAKS",  "HAT"))

    for num in range(folds_num):
        image_tensor = getTensor(num)

        for key in image_tensor:
            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_tensor[key]})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            results = {}
            # print('\n**%s**\n' % (key))
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                results[human_string] = score
                # print('%s (score = %.5f)' % (human_string, score))

            threashold = 0.27
            MAN, HAT, GLASS, MASK, TMGLASS = False, False, False, False, False
            if results["man"] >= results["woman"]:
                MAN = True
                gender = "man"
                xxx = results["woman"]
                if (results["man"] - results["woman"]) < 0.1:
                    xxx = threashold
            else:
                gender = "woman"
                xxx = results["man"]
                if (results["woman"] - results["man"]) < 0.1:
                    xxx = threashold

            if  results["hat"] >= xxx and results["hat"] > threashold:
                HAT = True

            if (results["glass"] >= xxx) and (results["glass"] > threashold) \
                and (results["glass"] > results["tmglass"]):
                GLASS = True

            if results["mask"] >= xxx and results["mask"] > threashold:
                MASK = True

            if (results["tmglass"] >= xxx) and (results["tmglass"] > threashold) \
                and (results["tmglass"] > results["glass"]):
                TMGLASS = True

            # test glass + tmglass
            GLASS_ALL = GLASS or TMGLASS

            # hgm_colors = {}
            # # if HAT or GLASS or MASK or TMGLASS:
            # if HAT or MASK or GLASS or TMGLASS:
            #     hgm_colors = getColor(folder_path, key, HAT, MASK, GLASS, TMGLASS)
            # # if hgm_colors:
            # #     for hgm_key in hgm_colors:
            # #         print(hgm_colors[hgm_key], hgm_key)

            # judge attributes true or false
            # predict_label = [TF_dict[not MAN], TF_dict[HAT], TF_dict[GLASS], TF_dict[MASK], TF_dict[TMGLASS]]
            # diff += np.abs(np.array(predict_label) - np.array(test_ground_truth[key]))
            predict_label = [TF_dict[not MAN], TF_dict[HAT], TF_dict[GLASS_ALL], TF_dict[MASK]]
            # test_labels = [man_label, hat_label, glass_label, mask_label, tmglass_label]
            tmp = test_ground_truth[key]
            g_all = tmp[2] + tmp[4]
            if g_all == 2:
                print('glass label error!')
            tmp_all = [tmp[0], tmp[1], g_all, tmp[3]]
            diff += np.abs(np.array(predict_label) - np.array(tmp_all))


            filename = "percent_results.txt"
            with open(filename, 'a+') as f:
                f.write('\n**%s**\n' % (key))
                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    f.write('%s (score = %.5f)\n' % (human_string, score))

            # filename = "results.txt"
            # with open(filename, 'a+') as f:
            #     f.write('\n%-13s' % (key))
            #     f.write('%-9s' % gender)
            #     f.write('%-7s' % yon((GLASS or TMGLASS)))

            #     f.write('%-7s' % yon(MASK))

            #     f.write('%-7s' % yon(HAT))

            filename = "rgzz.txt"
            with open(filename, 'a+') as f:
                f.write('%s' % (key))
                f.write(',')
                f.write('%s' % yon(MAN))
                f.write(',')
                f.write('%s' % yon((GLASS or TMGLASS)))
                f.write(',')
                f.write('%s' % yon(MASK))
                f.write(',')
                f.write('%s' % yon(HAT))
                f.write('\n')

    diff /= (1.0 * length)
    diff = 1 - diff
    percent = (diff * 100)[0]
    print('\ngender accuracy is ', percent[0])
    print('hat accuracy is ', percent[1])
    # print('sunglass accuracy is ', percent[2])
    # print('tmglass accuracy is ', percent[4])
    print('glass accuracy is ', percent[2])
    print('mask accuracy is ', percent[3])