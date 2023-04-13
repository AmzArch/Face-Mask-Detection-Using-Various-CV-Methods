#import packages
import os
import cv2 as cv
import numpy as np
import dlib
import argparse
import imutils


eye_pred="./eye_landmarks.dat"       #custom

# initialize dlib's face detector (HOG-based) and then load our
# trained shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(eye_pred)


print("started")

def rect_box(rect):
    '''
    # create a box around the detected face
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
        takes dlib coorinates and converts them into a np array
        INPUT: dlib shape
        OUTPUT: Np Array
    '''
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(48-36):
    	coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def gaussian_filter(kernel_size, sigma=1, mean=0):

    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
                       np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x**2+y**2)

    # lower normal part of gaussian
    norm = np.sqrt(2.0*np.pi*sigma**2) # Normalization constant

    # Calculating Gaussian filter
    gauss = np.exp(-((dst-mean)**2 / (2.0 * sigma**2))) / norm

    return gauss


def ord_dict(data):
    newDict = dict()
    for i in data:
        item = { (i) }
        newDict.update(item)

    return newDict


def loop_through_images(img):

    # load the input image, resize it, and convert it to grayscale
    image = cv.imread(img)
    # image = cv.resize(image, (0,0), fx = 1.2, fy = 1.2)   #make the image smaller
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    filt = gaussian_filter(3, sigma=4, mean=0)
    k=3
    edge = np.ones((k,k))*-1
    edge[1][1] = 8


    filtd = cv.filter2D(src=gray, ddepth=-1, kernel=edge)

    # detect faces in the grayscale image
    rects,scores,idx = detector.run(gray, 2, -.1)



    print("Number of faces detected: {}".format(len(rects)))
    print("rects", rects)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        print(" rect {}, score: {}, face_type:{}, i:{}".format(
            rect, scores[i], idx[i], i))


        shape = predictor(filtd, rect)
        print(shape)
        shape = to_nparray(shape)
        print(shape)
        clone = image.copy()

        # draw rectangle
        (x, y, w, h) = cv.boundingRect(np.array(shape))

        cv.rectangle(image,(x,y),(x+w,y+h), (0,0,255), 1)

    cv.imshow("Image", image)
    cv.waitKey(0)


# images_0
img_dir = "./images_0"
imgs = os.listdir(img_dir)
image_scores = dict()
for i in range(len(imgs)):
    img = f"{img_dir}/maksssksksss{i}.png"
    print("image path::::", img)
    # image_scores.update(loop_through_images(img))
    loop_through_images(img)
print(image_scores)
