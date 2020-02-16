#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .imgprocess import find_segment, clean_segment
import numpy as np
covert_text_box


# In[1]:


def get_points(img, thresh = 30):
    points = []
    group_n = 0
    #cut_img_1 = img3[:y2 - 10, x1 + 10:]
    for i in range(img.shape[1]):
        cur_piece = img[:, i]

        a = find_segment(cur_piece)
        a = clean_segment(a, thresh = thresh)
        if len(a) > 0:
            group_n = max(group_n, len(a))
            cur_point = [i, []]
            for k in a:
                cur_point[1].append(img.shape[0] - np.mean(k))

            points.append(cur_point)
    return points, group_n


# In[2]:


def find_group(p, points, thresh = 10):
    label = -1
    
    distance = [abs(k-p) for k in points]
    if min(distance) <= thresh:
        label = np.argmin(distance)
        
    return label


# In[3]:


def group_points(points, thresh = 100, min_points = 100):
    
    
    ## step1
    points_clean = {}

    max_label = 0
    for i in range(len(points)):
        points_clean[i] = []
        eles = points[i][1]
        if i == 0:
            for j in range(len(eles)):
                points_clean[i].append(j)

        else:
            for ele in eles:
                label = find_group(ele, points[i-1][1], thresh = thresh)
                if label != -1:
                    points_clean[i].append(label)
                else:
                    j += 1
                    points_clean[i].append(j)
        max_label = max(max_label, max(points_clean[i]))
    
    
    ## step 2
    points_group = {}
    for i in range(max_label + 1):
        points_group[i] = []

    for i in range(len(points)):
        eles = points[i][1]

        labels = points_clean[i]

        for j in range(len(labels)):
            points_group[labels[j]].append([i, eles[j]])
            
            
    ## step 3
    points_group_clean = {}
    j = 0
    for i in points_group:
        if len(points_group[i]) >= min_points:
            points_group_clean[j] = points_group[i]
            j += 1
            
    return points_group_clean, max_label


# In[ ]:





# In[ ]:





# In[ ]:




