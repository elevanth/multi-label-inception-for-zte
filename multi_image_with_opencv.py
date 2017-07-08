import tensorflow as tf
import numpy as np
from numpy import linalg as LA
import colorsys
import cv2
import os
import sys

MODE = "test"
folder_path = sys.argv[1]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("labels.txt")]

files = os.listdir(folder_path)
length = len(files)
batch_size = 128
folds_num = int(length / batch_size) + 1

image_tensor = {}

# def avg(img):
#     avg_color_per_row = np.average(img, axis=0)
#     avg_color = np.average(avg_color_per_row, axis=0)
#     # float to int
#     ans = [int(i) for i in avg_color]
#     return ans

def getIndex(idx, shape):
    return np.unravel_index(idx, shape)

def judge(lst):
    RGB = 30
    HSV = 30
    ths = RGB
    lmax = np.amax(lst)
    lmin = np.amin(lst)
    if (lmax-lmin) > ths:
        return False
    else:
        return True

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def getHSV(roi):
# def getHSV(bgr, GLASS=False, TMGLASS=False):
    dist = 360
    h = -1
    img_color = "hsv_init"
    REAL_COLOR = {}
    REAL_COLOR["BLACK"] = [0, 0, 0]
    # REAL_COLOR["DARK_RED    "] = [128, 0, 0]
    REAL_COLOR["RED1"] = [0, 1, 1]
    REAL_COLOR["RED2"] = [360, 1, 1]
    # REAL_COLOR["DARK_GREEN  "] = [0, 128, 0]
    # REAL_COLOR["DARK_YELLOW "] = [128, 128, 0]
    REAL_COLOR["GREEN"] = [120, 1, 1]
    REAL_COLOR["YELLOW"] = [60, 1, 1]
    # REAL_COLOR["DARK_BLUE   "] = [0, 0, 128]
    # REAL_COLOR["DARK_CARMINE"] = [128, 0, 128]
    # REAL_COLOR["DARK_CYAN   "] = [0, 128, 128]
    REAL_COLOR["GRAY"] = [0, 0, 0.5]
    # REAL_COLOR["GRAY        "] = [192, 192, 192]
    REAL_COLOR["BLUE"] = [240, 1, 1]
    REAL_COLOR["CARMINE"] = [300, 1, 1]
    REAL_COLOR["CYAN"] = [180, 1, 1]
    REAL_COLOR["WHITE"] = [0, 0, 1]
    # REAL_COLOR["ORANGE"] = [30, 1, 1]
    # REAL_COLOR["PRUSSIAN"] = [210, 1, 1]
    # REAL_COLOR["PURPLE"] = [270, 1, 1]
    # REAL_COLOR["PINK"] = [330, 1, 1]

    # bgr solution
    hist = cv2.calcHist([roi], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    sortedIndex = np.argsort(hist.flatten())
    bgr_list = [getIndex(sortedIndex[-i], hist.shape) for i in range(1,6)]
    sum_list = [(t[0]+t[1]+t[2]) for t in bgr_list]
    tag = judge(sum_list)
    while not tag:
        max_idx = np.array(sum_list).argmax()
        del sum_list[max_idx], bgr_list[max_idx]
        min_idx = np.array(sum_list).argmin()
        del bgr_list[min_idx], sum_list[min_idx]
        tag = judge(sum_list)
    if len(bgr_list) == 1:
        bgr_mean = getIndex(sortedIndex[-1], hist.shape)
    else:
        bgr_mean = np.mean(bgr_list, axis=0)
    if MODE == "valid":
        print(bgr_list, "\n", bgr_mean)
    # bgr = [i/255. for i in bgr_mean]
    # print(bgr)
    # r = bgr[2]
    # g = bgr[1]
    # b = bgr[0]
    # print(r,g,b)
    # hsv = colorsys.rgb_to_hsv(r, g, b)
    # print(hsv)
    # hue = hsv[0] * 360
    # sat = hsv[1]
    # val = hsv[2]
    hue, sat, val = rgb2hsv(bgr_mean[2], bgr_mean[1], bgr_mean[0])
    if MODE == "valid":
        print("hue",hue,"sat",sat,"val",val)
    '''
    # hsv solution
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [180, 100, 100], [0, 180, 0, 100, 0, 100])
    # max_index = np.unravel_index(hist.argmax(), hist.shape)
    sortedIndex = np.argsort(hist.flatten())
    hsv_list = [getIndex(sortedIndex[-i], hist.shape) for i in range(1,11)]
    hue_list = [t[0] for t in hsv_list]
    tag = False
    while not tag:
        max_idx = np.array(hue_list).argmax()
        min_idx = np.array(hue_list).argmin()
        del hue_list[max_idx], hue_list[min_idx]
        del hsv_list[max_idx], hsv_list[min_idx]
        tag = judge(hue_list)
    print("hsv_list length ", len(hsv_list))
    [hue, sat, val] = np.mean(hsv_list, axis=0)
    hue *= 2
    sat /= 100
    val /= 100
    print("hue",hue,"sat",sat,"val",val)
    '''
    # if val < 0.45:
    #     img_color = "BLACK"
    #     return img_color

    for key in REAL_COLOR:
        tmp = abs(hue - REAL_COLOR[key][0])
        if tmp < dist:
            dist = tmp
            real_h = REAL_COLOR[key][0]
            img_color = key
    if val < 0.4 and sat > 0.5:
        img_color = "DARK_" + img_color
    elif sat < 0.25 and val > 0.5:
        img_color = "BRIGHT_" + img_color
        
    if (real_h == 0) or (real_h == 360):
        if sat >0.7:
            img_color = "RED1"
            if val < 0.4 and sat > 0.5:
                img_color = "DARK_" + img_color
            elif sat < 0.25 and val > 0.5:
                img_color = "BRIGHT_" + img_color
        elif val > 0.7:
            img_color = "WHITE"
        elif val < 0.3:
            img_color = "BLACK"
        else:
            img_color = "GRAY"
    
    return img_color

    # if GLASS and TMGLASS:
    #     img_color = "error"
    # elif GLASS:
    #     img_color = "DARK"
    # elif TMGLASS:
    #     img_color = "BRIGHT"
    # else:
    #     [r, g, b] = np.array(bgr[::-1]) / 255
    #     # print([r, g, b])
    #     hsv = colorsys.rgb_to_hsv(r, g, b)
    #     # print(hsv)
    #     for key in REAL_COLOR: 
    #         tmp = abs(hsv[0] * 360 - REAL_COLOR[key][0])
    #         # print(key, tmp)
    #         if tmp < dist:
    #             dist = tmp
    #             h = REAL_COLOR[key][0]
    #             img_color = key
    #     if h == 0:
    #         if hsv[2] > 0.7:
    #             img_color = "BLACK"
    #         elif hsv[2] < 0.3:
    #             img_color = "WHITE"
    #         else:
    #             img_color = "GRAY"
    # # print(img_color)
    # return img_color

def getColor(folder_path, image_path, HAT, MASK, GLASS, TMGLASS):
    MEAN_HAT = [24, 13, 40, 28]
    # MEAN_GLASS = [31, 45, 32, 20]
    MEAN_MASK = [32, 70, 30, 27]

    colors = {}
    hat_color = "gC_init"
    glass_color = "gC_init"
    mask_color = "gC_init"

    real_image_path = folder_path + "\\" + image_path
    face_cascade = cv2.CascadeClassifier('opencv-haar-xml/haarcascade_frontalface_default.xml')

    img = cv2.imread(real_image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
    
    ori = cv2.imread(real_image_path)
    if isinstance(faces, tuple):
        if not faces:
            print("no face is found! Use default paramters.")
            hat_roi = ori[MEAN_HAT[1]:(MEAN_HAT[1]+MEAN_HAT[3]), MEAN_HAT[0]:(MEAN_HAT[0]+MEAN_HAT[2])]
            # glass_roi = img[MEAN_GLASS[1]:(MEAN_GLASS[1]+MEAN_GLASS[3]), MEAN_GLASS[0]:(MEAN_GLASS[0]+MEAN_GLASS[2])]
            mask_roi = ori[MEAN_MASK[1]:(MEAN_MASK[1]+MEAN_MASK[3]), MEAN_MASK[0]:(MEAN_MASK[0]+MEAN_MASK[2])]
        else:
            # print('[(x,y,w,h] is ', x, y, w, h)
            hat_roi = ori[max(int(y - h*3/11), 0) : int(y + h/11), int(x + w/10) : int(x + w*9/10)]
            # glass_roi = img[int(y + h*2/11) : int(y + 0.5*h), int(x + w*2/10) : int(x + w*8/10)]
            mask_roi = ori[int(y + h*6/11) : int(y + h*10/11), int(x + w*2.5/11) : int(x + w*8/11)]
    else:
        if not faces.any():
            print("no face is found! Use default paramters.")
            hat_roi = ori[MEAN_HAT[1]:(MEAN_HAT[1]+MEAN_HAT[3]), MEAN_HAT[0]:(MEAN_HAT[0]+MEAN_HAT[2])]
            # glass_roi = img[MEAN_GLASS[1]:(MEAN_GLASS[1]+MEAN_GLASS[3]), MEAN_GLASS[0]:(MEAN_GLASS[0]+MEAN_GLASS[2])]
            mask_roi = ori[MEAN_MASK[1]:(MEAN_MASK[1]+MEAN_MASK[3]), MEAN_MASK[0]:(MEAN_MASK[0]+MEAN_MASK[2])]
        else:
            # print('[(x,y,w,h] is ', x, y, w, h)
            hat_roi = ori[max(int(y - h*3/11), 0) : int(y + h/11), int(x + w/10) : int(x + w*9/10)]
            # glass_roi = img[int(y + h*2/11) : int(y + 0.5*h), int(x + w*2/10) : int(x + w*8/10)]
            mask_roi = ori[int(y + h*6/11) : int(y + h*10/11), int(x + w*2.5/11) : int(x + w*8/11)]
        
    if HAT:
        if MODE == "valid":
            hat_name = image_path + "hat.jpg"
            cv2.imwrite(hat_name, hat_roi)
            print("hat hsv is")
        hat_color = getHSV(hat_roi)
        # hat_color = avg(hat_roi)
    # if GLASS:
    #     glass_color = avg(glass_roi)
    # if TMGLASS:
    #     glass_color = avg(glass_roi)
    if MASK:
        if MODE == "valid":
            mask_name = image_path + "mask.jpg"
            cv2.imwrite(mask_name, mask_roi)
            print("mask hsv is")
        mask_color = getHSV(mask_roi)
        # mask_color = avg(mask_roi)
    if GLASS and TMGLASS:
        glass_color = "error"
    elif GLASS:
        glass_color = "DARK"
    elif TMGLASS:
        glass_color = "BRIGHT"
        
    colors['hat_color'] = hat_color
    colors['glass_color'] = glass_color
    colors['mask_color'] = mask_color

    return colors

'''
def recg_rgb(bgr, GLASS=False):
    dist = 400
    img_color = None
    REAL_COLOR = {}
    REAL_COLOR["BLACK"] = [0, 0, 0]
    # REAL_COLOR["DARK_RED    "] = [128, 0, 0]
    REAL_COLOR["RED"] = [255, 0, 0]
    # REAL_COLOR["DARK_GREEN  "] = [0, 128, 0]
    # REAL_COLOR["DARK_YELLOW "] = [128, 128, 0]
    REAL_COLOR["GREEN"] = [0, 255, 0]
    REAL_COLOR["YELLOW"] = [255, 255, 0]
    # REAL_COLOR["DARK_BLUE   "] = [0, 0, 128]
    # REAL_COLOR["DARK_CARMINE"] = [128, 0, 128]
    # REAL_COLOR["DARK_CYAN   "] = [0, 128, 128]
    REAL_COLOR["GRAY"] = [128, 128, 128]
    # REAL_COLOR["GRAY        "] = [192, 192, 192]
    REAL_COLOR["BLUE"] = [0, 0, 255]
    REAL_COLOR["CARMINE"] = [255, 0, 255]
    REAL_COLOR["CYAN"] = [0, 255, 255]
    REAL_COLOR["WHITE"] = [255, 255, 255]

    if GLASS:
        img_color = "DARK"
    else:
        rgb = bgr[::-1]
        for key in REAL_COLOR: 
            tmp = LA.norm(np.array(rgb) - np.array(REAL_COLOR[key]), 1)
            # print(key, tmp)
            if tmp < dist:
                dist = tmp
                img_color = key
    return img_color
'''

# print yes or no
def yon(YN):
    if YN:
        return "yes"
    else:
        return "no"

def getTensor(num):
    img_tensor = {}
    batch_list = files[num*batch_size : min((num+1)*batch_size, length)]
    for image_path in batch_list:
        real_image_path = folder_path + "\\" + image_path
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
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    filename = "results.txt"    
    with open(filename, 'a+') as f:
        f.write('\n%-13s%-9s%-7s%-9s%-7s%-15s%-7s%-15s' \
            % ("FILES", "GENDER", "GLASS", "COLOR", "MAKS", "COLOR", "HAT", "COLOR"))
    
    for num in range(folds_num):
        image_tensor = getTensor(num)

        for key in image_tensor:
            predictions = sess.run(softmax_tensor, \
                    {'DecodeJpeg/contents:0': image_tensor[key]})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            results = {}
            print('\n**%s**\n' % (key))
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                results[human_string] = score
                print('%s (score = %.5f)' % (human_string, score))

            threashold = 0.27
            HAT, GLASS, MASK, TMGLASS = False, False, False, False
            if results["man"] >= results["woman"]:
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

            hgm_colors = {}
            # if HAT or GLASS or MASK or TMGLASS:
            if HAT or MASK or GLASS or TMGLASS:
                hgm_colors = getColor(folder_path, key, HAT, MASK, GLASS, TMGLASS)
            # if hgm_colors:
            #     for hgm_key in hgm_colors:
            #         print(hgm_colors[hgm_key], hgm_key)
            

            filename = "percent_results.txt"    
            with open(filename, 'a+') as f:
                f.write('\n**%s**\n' % (key))
                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    f.write('%s (score = %.5f)\n' % (human_string, score))

            filename = "results.txt"    
            with open(filename, 'a+') as f:
                f.write('\n%-13s' % (key))
                f.write('%-9s' % gender)
                f.write('%-7s' % yon((GLASS or TMGLASS)))
                if GLASS or TMGLASS:
                    f.write('%-9s' % (hgm_colors['glass_color']))
                else:
                    f.write('%-9s' % ("None"))
                
                f.write('%-7s' % yon(MASK))
                if MASK:
                    f.write('%-15s' % (hgm_colors['mask_color']))
                else:
                    f.write('%-15s' % ("None"))
                
                f.write('%-7s' % yon(HAT))
                if HAT:
                    f.write('%-15s' % (hgm_colors['hat_color']))
                else:
                    f.write('%-15s' % ("None"))