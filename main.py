# cmake
# boost
# cv
# XQuartz
# dlib
# mtcnn
# Tensorflow

import os
import cv2 as cv
import numpy as np
import dlib
import mtcnn
import matplotlib.pyplot as plt


# load constellations
# shape_predictor="./eye_constellations.dat"             #eyes only
shape_predictor="./feature_constellations.dat"      #all features


# initialize dlib's feature detector (HOG + constellation ) and then create
# the facial landmark predictor
feature_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

def ord_dict(data):
    newDict = dict()
    for i in data:
        item = { (i) }
        newDict.update(item)

    return newDict

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = ord_dict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


# get box coords
def rect_box(rect):
    '''
    # get coords around the detected face
        INPUT: dlib face region
        OUTPUT: 4 part tuple (x, y, w, h)
    '''
    # take a bounding predicted by dlib and convert it
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def to_nparray(shape, dtype="int"):
    '''
        #takes dlib coorinates and converts them into a np array
        INPUT: dlib shape
        OUTPUT: Np Array
    '''
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(68):
    	coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def gaussian_filter(kernel_size, sigma=1, m=0):
    '''
        # creates a gaussian filter
        INPUT: size of kernel (int), factor, mean
        OUTPUT: kernel, of size kernel_size
    '''
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)

    # lower normal part of gaussian
    norm = np.sqrt(2.0*np.pi*sigma**2) # Normalization constant

    # Calculating Gaussian filter
    gauss = np.exp(-((dst-m)**2 / (2.0 * sigma**2))) / norm

    return gauss


# find constellation of features
def detect_features(img):
    '''
        # determines if face features exist in image
        INPUT: image
        OUTPUT: filtered array
    '''

    # load the input image, resize it, and convert it to grayscale
    image = cv.imread(img)
    # image = cv.resize(image, (0,0), fx = 1.2, fy = 1.2)   #make the image smaller
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    gauss = gaussian_filter(3, sigma=5, m=0)
    k=3
    edge = np.diag(np.zeros(k-1)+-1,-1) + np.diag(np.zeros(k-1)+-1,1)
    edge[1][1] = 6

    # apply filters
    edgd = cv.filter2D(src=gray, ddepth=-1, kernel=edge)    # accentuate edges
    filtd = cv.filter2D(src=edgd, ddepth=-1, kernel=gauss)  # smooth out low delta sections

    # detect faces in the filtered grayscale image
    rects,scores,idk = feature_detector.run(filtd, 2, .25)

    # if no feature constellations are found, mask is present
    if ( len(rects) < 1 ):
        cv.putText(image, "Wearing Mask.", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("Image", image)
        cv.waitKey(0)
        return 1

    # if image contain feature constellations, count each
    else:
        # loop over the face detections
        for (i, rect) in enumerate(rects):

            # convert feature constellation into x,y coords
            shape = predictor(image, rect)
            shape = to_nparray(shape)

            # highlight exposed parts
            for (x, y) in shape[48:68]:                         # draw mouth
                cv.circle(image, (x, y), 1, (0, 0, 255), -1)
            for (x, y) in shape[27:35]:                         # draw nose
                cv.circle(image, (x, y), 1, (0, 0, 255), -1)
            cv.putText(image, "No Mask.", (5, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.imshow("Image", image)
            cv.waitKey(0)

            return 0



def draw_facebox(filename, result_list):
    # load the image
    data = plt.imread(filename)
    # plot the image
    plt.imshow(data)
    # get the context for drawing boxes
    ax = plt.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = plt.Rectangle((x, y), width, height, fill=False, color='green')
        # draw the box
        ax.add_patch(rect)
        # show the plot
        plt.show()

# determine if object is present or face
def detect_object(img):

    # load image from file
    # pixels = cv.imread(img)
    #
    # # distinguish objects from faces
    # objects = ob_detector.detect_faces(pixels)
    # print(objects)
    # print(pixels)
    # if ( len(objects) < 1 ):
    #     return 0
    # else:
    #     # for face in objects:
    #     #     image = face["box"][]
    #     # features = detect_features(img)
    #     # draw an image with detected objects
    #     draw_facebox(img, objects)
    #     # if ( features ):
    #     #     return 1
    #     # else:
    #     #     return 0

    image = cv.imread(img)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gauss = gaussian_filter(3, sigma=2, m=0)
    k=3
    edge = np.diag(np.zeros(k-1)+-1,-1) + np.diag(np.zeros(k-1)+-1,1)
    edge[1][1] = 6

    # apply filters
    edgd = cv.filter2D(src=gray, ddepth=-1, kernel=edge)    # accentuate edges
    # filtd = cv.filter2D(src=gray, ddepth=-1, kernel=gauss)  # smooth out low delta sections


    # Load image

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.8

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(edgd)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv.drawKeypoints(edgd, keypoints, blank, (0, 0, 255),
                              cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv.putText(edgd, text, (20, 550),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs
    cv.imshow("Filtering Circular Blobs Only", edgd)
    cv.waitKey(0)
    cv.destroyAllWindows()


# loop through all images in "images" folder
# img_dir = "./images" # images_0
img_dir = "./images_mixed"
imgs = os.listdir(img_dir)
# ttl_no_masks = 480
# ttl_masks = 481
no_face     = 0
no_masks    = 0
masks       = 0
for i in imgs:
    img = f"{img_dir}/{i}"
    print("image path::::", img)
    result = detect_object(img)
    # if ( result < 0):
    #     no_face += 1
    # elif ( result == 0 ):
    #     no_masks += 1
    # else:
    #     masks += 1
print("Results: No Faces {}; No Masks {}; Masks {}".format(no_face,no_masks,masks))
# print("Accuracy: No Masks{}; Masks{}".format(no_masks/ttl_no_masks, masks/ttl_masks))
