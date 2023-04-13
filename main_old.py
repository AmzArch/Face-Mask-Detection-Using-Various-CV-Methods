import numpy as np
import matplotlib as plt
from matplotlib import pyplot

# from cv2 import imgproc, features2d         ## https://docs.opencv.org/3.4/d1/dfb/intro.html
import cv2 as cv


#########
#       https://www.analyticsvidhya.com/blog/2021/06/feature-detection-description-and-matching-of-images-using-opencv/
#########

# // Command to be typed for running the sample
# ./sampleDetectLandmarks -file=trained_model.dat -face_cascade=lbpcascadefrontalface.xml -image=/path_to_image/image.jpg

# 4 steps
# (1) img preprocessing
# (2) face detection
# (3) feature detection
# (4) mask detection

# cascade_name =
# image =
# filename =

# Create cascade object and import image
# CascadeClassifier face_cascade;
# face_cascade.load(cascade_name);
# Mat img = imread(image);                                # import image
# Ptr<Facemark> facemark = createFacemarkKazemi());
# facemark->loadModel(filename);                          #import model
# cout<<"Loaded model"<<endl;

#
#
# vector<Rect> faces;
# resize(img,img,Size(460,460),0,0,INTER_LINEAR_EXACT);
# Mat gray;
# std::vector<Rect> faces;
# if(img.channels()>1){
#     cvtColor(img.getMat(),gray,COLOR_BGR2GRAY);
# }
# else{
#     gray = img.getMat().clone();
# }
# equalizeHist( gray, gray );
# face_cascade.detectMultiScale( gray, faces, 1.1, 3,0, Size(30, 30) );


# vector< vector<Point2f> > shapes;
# if (facemark->fit(img,faces,shapes))
# {
#     for ( size_t i = 0; i < faces.size(); i++ )
#     {
#         cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
#     }
#     for (unsigned long i=0;i<faces.size();i++){
#         for(unsigned long k=0;k<shapes[i].size();k++)
#             cv::circle(img,shapes[i][k],5,cv::Scalar(0,0,255),FILLED);
#     }
#     namedWindow("Detected_shape");
#     imshow("Detected_shape",img);
#     waitKey(0);
# }

file = "./images/keck.JPEG"
# with open(file,'r') as f:
#     print(f)

# img = cv.imread(file)
# img_gray = cv.imread(file,0)
# img_gray = cv.IMREAD_GRAYSCALE(file)



# tst = img_gray - 20
# tst = img_gray  20
# tst = img_gray
# tst.shape
# high_pass = np.diag(np.ones(tst.shape[1] -1), -1) + np.diag(np.ones(tst.shape[1])*-2) + np.diag(np.ones(tst.shape[1] -1), 1)

lft_brdr = 300
rit_brdr = 670
top_brdr = 90
btm_brdr = 460

# Cropping an image
# cropd = np.array(cropd)
# cropd.shape
# cropd.reshape(cropd.shape[1],1)
# high_pass = np.diag(np.ones(cropd.shape[1] -1)*-.2, -1) + np.diag(np.ones(cropd.shape[1])*-.5) + np.diag(np.ones(cropd.shape[1] -1), 1)
# high_pass
# orig_pass = high_pass * cropd

#####
#    HARRIS CORNER DETECTION
#####

no_mask_img = "./images/keck.JPEG"
mask_img = "./images/keck_mask_1.JPEG"
bad_mask_img = "./images/keck_mask_bad.JPEG"

imput_img = "./images/keck.JPEG"
land_img = "./images/scape.JPG"
ori = cv.imread(imput_img)
image = cv.imread(imput_img)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# gray = np.float32(gray)
# dst = cv.cornerHarris(gray,2,3,0.04)
# dst = cv.dilate(dst,None)
# image[dst>0.01*dst.max()]=[0,0,255]
# cv.imshow('Original',ori)
# cv.imshow('Harris',image)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()



#####
#    SHI-TOMASI CORNER DETECTION
#####
def get_SHIT_features(file, name="Shi-Tomasi"):
    '''
        Desc:   Utilizes the SHI Tomasi algorithm to identify corners on the image
        Input:  file name -> STRING
        Return: None
    '''
    # pass`
    img = cv.imread(file)
    ori = cv.imread(file)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray,20,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    cv.imshow('Original', ori)
    cv.imshow('Shi-Tomasi', img)

    if cv.waitKey(0) == 27:
        cv.destroyAllWindows()

# get_SHIT_features(bad_mask_img)
# get_SHIT_features(no_mask_img, "no Mask")


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

get_SIFT_features(no_mask_img)



# SURF --> COULDN'T USE BECAUSE:: NONFREE
# surf = cv.xfeatures2d.SURF_create(400)
# kp, des = surf.detectAndCompute(image,None)
# img2 = cv.drawKeypoints(image,kp,None,(255,0,0),4)
# cv.imshow('Original', ori)
# cv.imshow('SURF', img2)


#BLOB DETECTION
def get_BLOB_features(img):
    '''
        Desc:   Utilizes the BLOB algorithm to identify keypoints on the image
        Input:  file name -> STRING
        Return: None
    '''
    ori = cv.imread(img)
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    detector = cv.SimpleBlobDetector_create()
    keypoints = detector.detect(img)
    img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('Original',ori)
    cv.imshow('BLOB',img_with_keypoints)

    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()



# HISTOGRAM OF GRADIENT --> NEED TO DOWNLOAD SciKit https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html
# from skimage.feature import hog
# img = image
# _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                     cells_per_block=(1, 1), visualize=True, multichannel=True)
# cv.imshow('Original', ori)
# cv.imshow('HoG', hog_image)


# BRIEF (Binary Robust Independend Elementary Features )
def get_BRIEF_features(img):
    '''
        Desc:   Utilizes the BRIEF algorithm to identify keypoints on the image
        Input:  file name -> STRING
        Return: None
    '''
    land_ori = cv.imread(img)
    img = cv.imread(img, 0)
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp = star.detect(img,None)
    kp, des = brief.compute(img, kp)
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    cv.imshow('Original', land_ori)
    cv.imshow('BRIEF', img2)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


# ORB
def get_ORB_features(img):
    '''
        Desc:   Utilizes the ORB algorithm to identify keypoints on the image
        Input:  file name -> STRING
        Return: None
    '''
    land_ori = cv.imread(img)
    img = cv.imread(img, 0)
    orb = cv.ORB_create(nfeatures=200)
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    cv.imshow('Original', land_ori)
    cv.imshow('ORB', img2)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()


# # Feature matching
# img1 = cv.imread('/content/det1.jpg', 0)
# img2 = cv.imread('/content/88.jpg', 0)
# orb = cv.ORB_create(nfeatures=500)
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des = orb.detectAndCompute(img2, None)
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None)
# cv.imshow('original image', img1)
# cv.imshow('test image', img2)
# cv.imshow('Matches', match_img)
# cv.waitKey()


# Output img with window name as 'image'
# cv.imshow('keck', img)
# cv.imshow('keck', cropd)
# cv.imshow('keck', orig_pass)

# Maintain output window utill
# user presses a key
# cv.waitKey(0)

# Destroying present windows on screen
# cv.destroyAllWindows()
