#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import pytesseract


# In[ ]:





# In[2]:


def get_img_ocr(image, box, config_number = 3, ratio = 2):
    #img = Image.open(imgfile)
    #custom_config = r'--oem 3 --psm 6'
    #####img = cv2.imread(imgfile)
    
    # default is 3: fully automatic
    # support 1-12
    # may use 1, 6, 11, 12
    custom_config =  r'--oem 3 --psm ' + str(config_number)
    
    img = image.copy()
    x1, y1, x2, y2 = box
    img = img[y1-10: y2+10, x1-10: x2+10, :]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    height, width = img.shape[:2]
    
    img_1 = cv2.resize(img,(int(ratio*width), int(ratio*height)), interpolation = cv2.INTER_CUBIC)
    img_array = np.array(img_1)
    img_array = img_array[:, :, np.newaxis]
    H, W, _ = img_array.shape
    text = pytesseract.image_to_string(img_1, config = custom_config)
    bbox = pytesseract.image_to_boxes(img_1, config = custom_config) # (x1, y1, x2, y2)
    #img.close()
    ######imgname = path + '/scanned.png'
    ######pix.writePNG(imgname)
    return bbox, text, H, W


# In[3]:


def get_axis_label(img, box, xaxis, yaxis, tol_main = 50, tol_minor = 10):
    r = min(1900/img.shape[0], 1500/img.shape[1])
    r = round(r, 2)

    x_label = {}
    y_label = {}

    for i in range(len(box)):
        x1, y1, x2, y2 = box[i]
        bbox, text, H, W = get_img_ocr(img, box[i], config_number = 6, ratio = r)

        if abs(y1 - yaxis) <= tol_main and x2 + tol_minor >= xaxis: #x2
            x_label[text] = np.mean([x1, x2])

        if abs(x2 -  xaxis) <= tol_main and y1 <= yaxis + tol_minor: #y1
            y_label[text] = img.shape[0] - np.mean([y1, y2])

    x_label = sorted(x_label.items(), key = lambda x: x[1])
    y_label = sorted(y_label.items(), key = lambda y: y[1])
    
    x_label = dict(x_label)
    y_label = dict(y_label)
    
    return x_label, y_label


# In[4]:


def shift_label(label):
    label_shift = {}
    eles = list(label.keys())
    
    for i in range(len(eles)):
        label_shift[eles[i]] = label[eles[i]] - label[eles[0]] 
    
    return label_shift


# In[5]:


def get_label_ratio(label):
    
    eles = list(label.keys())
    r = []
    for i in range(1, len(eles)):
        _r = (float(eles[i]) - float(eles[0])) /(label[eles[i]] - label[eles[0]])
        r.append(_r)
        
    return np.mean(r), eles[0], r


# In[6]:


def convert_axis(points, label):
    
    r, start, _ = get_label_ratio(label)
    points_conv = {}
    
    for i in points:
        points_conv[i] = []
        for point in points[i]:
            point = [k*r + float(start) for k in point]
            points_conv[i].append(point)
                
    return  points_conv


# In[ ]:




