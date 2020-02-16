#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
import cv2
import numpy as np


# In[ ]:





# In[3]:


def preprocess(img, block = 11, C = 5, dilate_iter = 1, erode_iter = 10):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = gray.astype(np.uint8) #############################
    # smooth the image to avoid noises
    gray = cv2.medianBlur(gray,5)

    # Python: cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst
    #src – Source 8-bit single-channel image.
    #dst – Destination image of the same size and the same type as src .
    #maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    #adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . See the details below.
    #thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    #blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
    #C – Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.


    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block,C)
    thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = dilate_iter)
    thresh = cv2.erode(thresh,None,iterations = erode_iter)
    thresh = cv2.dilate(thresh,None,iterations = dilate_iter)
    #thresh = cv2.erode(thresh,None,iterations = erode_iter)
    #thresh = cv2.dilate(thresh,None,iterations = dilate_iter)
    
    
    pre = ~thresh
    
    return pre


# In[8]:


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):

    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] #[k[0] for k in boxes] 
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes 
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int/(area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes


# In[ ]:





# In[4]:


## group close lines
def grouplines(cols, min_sep, method = 'min'):
    grouped = []
    i = 0
    while i < len(cols):
        checked = [i]
        cur_group = [cols[i]]
        j = i+1
        while j < len(cols):
            if cols[j] - cols[i] <= min_sep:
                cur_group.append(cols[j])
                checked.append(j)
                i = j
                j = j+1
            else:
                j +=1
        grouped.append(cur_group)
        i = checked[-1] + 1

    cols_clean = []
    for i in range(len(grouped)):
        #col = np.mean(grouped[i])
        #col = np.max(grouped[i])
        if method == 'min':
            col = np.min(grouped[i])
            
        if method == 'max':
            col = np.max(grouped[i])
            
        if method == 'mean':
            col = np.mean(grouped[i])
            
        if method == 'mode':
            col = mode(grouped[i])[0][0]
        
        
        cols_clean.append(int(col))
        
    return cols_clean


# In[5]:


## use hough transform to get lines
def get_hough_lines(img, thresh = 50, minline = 100, maxgap = 10, resmax = 20, hor_tol = 20, ver_tol = 15, method = 'mean'):
    img_t = img.copy()
    gray = cv2.cvtColor(img_t,cv2.COLOR_BGR2GRAY)
    ##gray = cv2.medianBlur(gray,5)
    ##gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)
    edges = cv2.Canny(gray,thresh,2*thresh,apertureSize = 3) #50, 150
    #minLineLength = 100
    #maxLineGap = 10 # 10, 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,resmax,minline,maxgap) # 50, 20, 1
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img_t,(x1,y1),(x2,y2),(0,255,0),1)
            
    hor_lines = {}
    ver_lines = {}
    for line in lines:
        for ele in line:
            # hor
            if ele[1] == ele[3]:
                if ele[1] not in hor_lines:
                    hor_lines[ele[1]] = 1
                else:
                    hor_lines[ele[1]] += 1
            if ele[0] == ele[2]:
                if ele[0] not in ver_lines:
                    ver_lines[ele[0]] = 1
                else:
                    ver_lines[ele[0]] += 1
                    
                    
    hor_clean = []
    ver_clean = []
    for line in lines:
        for ele in line:
            # hor
            if ele[1] == ele[3]:
                if ele[1] not in hor_clean and ( abs(ele[2] - ele[0]) > hor_tol and hor_lines[ele[1]] >= 1): # 20 and >= 1
                    hor_clean.append(ele[1])
            if ele[0] == ele[2]:
                if ele[0] not in ver_clean and ( abs(ele[3] - ele[1]) > ver_tol and ver_lines[ele[0]] >= 1): # 15 for test5, 10 for short
                    ver_clean.append(ele[0])
    hor_clean = sorted(hor_clean)
    hor_clean = grouplines(hor_clean, 10, method = method)
    ver_clean = sorted(ver_clean)
    ver_clean = grouplines(ver_clean, 10, method = method)
    
    return hor_clean, ver_clean


# In[7]:


def findchars(textpre, w_min = 5, w_max = 50, h_min = 30, h_max = None):
    
    if h_max is None:
        h_max = 0.8*textpre.shape[0]
        
    contours1 = cv2.findContours(textpre, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE
    contours1 = contours1[0] if len(contours1) == 2 else contours1[1]
    contours2 = cv2.findContours(~textpre, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE
    contours2 = contours2[0] if len(contours2) == 2 else contours2[1]
    contours = contours1 + contours2
    
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w_min<=w<=w_max and h_min<=h<=h_max:
            boxes.append([x, y, x+w, y+h])
   
    return boxes


# In[ ]:





# In[9]:


def covert_text_box(img, box, tol = 10):
    
    for b in box:
        x1, y1, x2, y2 = b
        img[y1 - tol:y2 + tol, x1 - tol:x2 + tol]  = 0
        
    return img


# In[10]:


def find_segment(line):
    allsegs = []
    i = 0
    
    while i < len(line):
        
        if line[i] == 255 and i == len(line):
            break
        elif line[i] == 255 and i != len(line):
            cur_seg = [i]
            j = i+1 

            while j < len(line):
                if line[j] == 255 and j == len(line):
                    cur_seg.append(len(line))
                    break
                elif line[j] == 255 and j != len(line):
                    j += 1
                    
                    if j == len(line):
                        cur_seg.append(len(line))
                        break
                else: 
                    cur_seg.append(j)
                    break
            
            #if cur_seg[-1] - cur_seg[0] >= thresh:
            allsegs.append(cur_seg)
            i = j 
        else:
            i += 1
    
    return allsegs


# In[11]:


def clean_segment(allsegs, thresh = 5):
    allsegs_clean = []
    for seg in allsegs:
        if seg[-1] - seg[0] >= thresh:
            allsegs_clean.append(seg)
    return allsegs_clean


# In[ ]:





# In[ ]:





# In[12]:


## visulize lines
def vis_lines(img, ver_line, hor_line):
    image = copy.deepcopy(img)

    # Remove vertical lines
    for c in ver_line:
        cv2.line(image, (c, 0), (c, image.shape[0]), (0,255,0), 6)

    for c in hor_line:
        cv2.line(image, (0, c), (image.shape[1], c), (255,0,0), 6)
     
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    ax.imshow(image)
    #return image


# In[13]:


## visulize boxes
def vis_boxes(boxes, img, x_tol = 0, y_tol = 0):
    
    t = img.copy()
    for box in boxes:
        cv2.rectangle(t,(box[0] - x_tol, box[1] - y_tol),(box[2] + x_tol, box[3] + y_tol),(0,255,0), 4)

    # Finally show the image
    #plt.imshow(t)
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    ax.imshow(t)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




