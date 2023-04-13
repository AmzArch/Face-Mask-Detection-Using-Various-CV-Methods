import numpy as np
import matplotlib as plt
from matplotlib import pyplot

# from cv2 import imgproc, features2d         ## https://docs.opencv.org/3.4/d1/dfb/intro.html
import cv2 as cv

file = "./keck_images/keck.JPEG"

lft_brdr = 300
rit_brdr = 670
top_brdr = 90
btm_brdr = 460

no_mask_img = "./keck_images/keck.JPEG"
mask_img = "./keck_images/keck_mask_1.JPEG"
bad_mask_img = "./keck_images/keck_mask_bad.JPEG"

imput_img = "./keck_images/keck.JPEG"
land_img = "./keck_images/scape.JPG"
ori = cv.imread(imput_img)
image = cv.imread(imput_img)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)


def img_walk_about(image):
    '''
        input: img array
        output: number of times targ is detected
    '''

    img = cv.imread(image)
    ori = cv.imread(image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #filter img
    cv.imshow("orig", gray)
    cv.waitKey(0)

    # Ensure array format is set
    brow    = np.array([[0,0,0,0],[1,1,1,1],[1,1,1,1],[0,0,0,0]])
    mouth   = np.array([[0,0,0,0],[1,1,1,1],[.5,.5,.5,.5],[1,1,1,1]])
    nose    = np.array([[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]])

    n   = 1

    img_height = gray.shape[0]
    img_width = gray.shape[1]

    filt = np.diag((np.ones(2)*-1),1) + np.diag((np.ones(2)*-1),-1)
    filt[1][1]=4

    count = 0
    brows =[]
    mouths =[]
    noses =[]
    procd_img = []
    # for h in range(img_height-n):
    #     for w in range(img_width-n):
    #         window = gray[h:h+n,w:w+n]
    #         filt = window*filt
    #         procd_img.append(filt)
    #         if (np.array_equal(window, mouth)):
    #             count += 1
    #             brows.append(np.array(window*kern/window.shape[0]**2))
    print('count', count)
    print('features', brows)
    procd_img = np.array(gray) * filt
    cv.imshow("procd", procd_img)
    cv.waitKey(0)


    features = [ ]
    # features = [ img for img in procd_img if img.all() < 150 ]
    return features

img_walk_about(no_mask_img)




# SIFT
def get_SIFT_features(img):
    '''
        Desc:   Utilizes the SIFT algorithm to identify keypoints on the image
        Input:  file name -> STRING
        Return: None
    '''
    # img = img[280:580,90:460]
    bad_mask = cv.imread(img)
    b_m_img = cv.imread(img)
    sift = cv.SIFT_create()
    gray = cv.cvtColor(b_m_img,cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray,None)
    img = cv.drawKeypoints(b_m_img,kp,gray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('Original', bad_mask)
    cv.imshow('SIFT',img)
    # cv.imshow('DES', des)
    # pyplot.plot(des)
    print(kp)
    print(des)
    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()

# get_SIFT_features(no_mask_img)
