#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.imgprocess import *
from utils.readaxis import *
from utils.curvepoints import *
import pickle

# In[ ]:





# In[ ]:





# In[48]:


## load image and resize for optimization
def read_img(filepath, optimal_size = 1000, show_img = False):
    img = plt.imread(filepath) #plt.imread('./curves/test2.jpg', 'CV_8UC1')
    h, w = img.shape[:2]
    r = np.ceil(max((optimal_size/h, optimal_size/w)))
    img = cv2.resize(img, (0,0), fx=r, fy=r, interpolation=cv2.INTER_CUBIC) # use CUBIC 
    
    if show_img:
        plt.imshow(img)
    
    return img


# In[49]:


def parse_axis_line(img, 
                  thresh = 1, 
                  minline = 100, 
                  maxgap = 10, 
                  resmax = 20, 
                  hor_tol = 200, 
                  ver_tol = 50, 
                  method = 'mean', 
                  show_line = False):
    
    ## default: (thresh = 50, minline = 100, maxgap = 10, resmax = 20, hor_tol = 20, ver_tol = 15, method = 'mean')
    hor_clean, ver_clean = get_hough_lines(img, thresh, minline, maxgap, resmax, hor_tol, ver_tol, method)
    yaxis = hor_clean[-1]
    xaxis = ver_clean[0]

    if show_line:
        img1 = vis_lines(img, [xaxis], [yaxis])
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        ax.imshow(img1)
        
    return xaxis, yaxis, hor_clean, ver_clean


# In[50]:


def parse_axis_label(img, show = False):
    
    img1 = preprocess(img, block = 11, C = 5, dilate_iter = 2, erode_iter = 15) #2, 15

    box = findchars(img1, w_min = 10, w_max = 2000, h_min = 30, h_max = 200) #w_min = 20, w_max = 2000, h_min = 20, h_max = 200

    probs = np.array([1] * len(box))
    box = non_max_suppression_fast(np.array(box), probs, overlap_thresh=0.1, max_boxes=300)
    box = sorted(box, key = lambda x: x[0])

    # vis_boxes(box, img, x_tol = 5, y_tol = 5)

    x_label, y_label = get_axis_label(img, box, xaxis, yaxis)
    x_label = shift_label(x_label)
    y_label = shift_label(y_label)

    return box, x_label, y_label


# In[57]:


def convert_curve_to_points(img, box, cut_tol = 10):
    img1 = preprocess(img, block = 11, C = 7, dilate_iter = 1, erode_iter = 15)
    img2 = covert_text_box(img1, box, tol = 30)

    cutimg = img2[:yaxis - cut_tol, xaxis + cut_tol:]

    points, group_n = get_points(cutimg, thresh = 30)
    points_group, max_label = group_points(points, thresh = 100, min_points = 100)

    return points_group, max_label


# In[ ]:





# In[ ]:





# In[58]:


def compare(img, points_conv):
    ## check final result, compare 2 graphs
    fig, ax = plt.subplots(2, 1, figsize = (10, 10))

    ax[0].imshow(img)

    #for i in points_group:

        #X = [k[0] for k in points_group[i]]
        #Y = [k[1] for k in points_group[i]]
        #ax[1].scatter(X, Y)


    ## plot real axis
    for i in points_conv:

        X = [k[0] for k in points_conv[i]]
        Y = [k[1] for k in points_conv[i]]

        plt.scatter(X, Y)
    
    


# In[ ]:





# ## test

# In[60]:

'''
filepath = './curves/test0.jpg'
img = read_img(filepath)
xaxis, yaxis, hor_clean, ver_clean = parse_axis_line(img)
# vis_lines(img, [xaxis], [yaxis])
box, x_label, y_label = parse_axis_label(img)
# vis_boxes(boxes, img)
points_group, max_label = convert_curve_to_points(img, box)
points_conv = convert_axis(points_group, y_label)
# compare(img, points_conv)


# In[61]:


points_conv.keys()


# In[62]:


compare(img, points_conv)


# In[ ]:
'''




# In[ ]:





# ## Script run

# In[ ]:


if __name__ == '__main__':
    filepath = './curves/test0.jpg'
    img = read_img(filepath)
    xaxis, yaxis, hor_clean, ver_clean = parse_axis_line(img)
    # vis_lines(img, [xaxis], [yaxis])
    box, x_label, y_label = parse_axis_label(img)
    # vis_boxes(boxes, img)
    points_group, max_label = convert_curve_to_points(img, box)
    points_conv = convert_axis(points_group, y_label)
    # compare(img, points_conv)
    print("%d lines are detected" % (len(points_conv)))
    with open('curve_data_points.pkl', 'wb') as f:
        pickle.dump(points_conv, f)



