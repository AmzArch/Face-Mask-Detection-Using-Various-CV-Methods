import cv2
import sys
import os
import keras
import tensorflow as tf
import numpy as np
import dlib

class faceMaskDetectionMethod():
    def __init__(self, desc, impl, img_dir):  # main function
        # Method description
        self.description = desc
        # Method implementation:
        # The function must take in a dictionary of {"image_name": cv_object} pair}, the expected class/result,
        #   and a dictionary of true/false positive/negative with their counts.
        # It must return the total number of images ran
        self.impl = impl
        # Image directory
        self.img_dir = img_dir
        # Metrics
        self.overall_accuracy = 0
        self.class_accuracy = {"without_mask": 0, "with_mask": 0}
        self.false_positive = 0
        self.false_negative = 0
        self.f1_score = 0

    def run(self):
        stats = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
        for key in self.class_accuracy:
            # Get images
            img_folder = os.path.join(self.img_dir, key)
            images = readAllImages(img_folder)
            num_images = self.impl(images, key, stats)
            if key == "without_mask":
                self.class_accuracy[key] = round((stats["true_negative"] / num_images), 2)
                self.false_positive = round((stats["false_positive"] / num_images), 2)
            elif key == "with_mask":
                self.class_accuracy[key] = round((stats["true_positive"] / num_images), 2)
                self.false_negative = round((stats["false_negative"] / num_images), 2)
        overall_accuracy = sum(self.class_accuracy.values()) / float(len(self.class_accuracy))
        self.overall_accuracy = overall_accuracy
        # Precision = True Positive / (True Positive + False Positive)
        # Recall = True Positive / (True Positive + False Negative)
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        precision = stats["true_positive"] / (stats["true_positive"] + stats["false_positive"])
        recall = stats["true_positive"] / (stats["true_positive"] + stats["false_negative"])
        self.f1_score = round((2 * (precision * recall) / (precision + recall)), 2)
    
    def printResult(self):
        print(f"Using {self.description} yields:")
        print(f"Overall accuracy: {self.overall_accuracy}")
        print(f"Class accuracy - Without mask: {self.class_accuracy['without_mask']}")
        print(f"Class accuracy - With mask: {self.class_accuracy['with_mask']}")
        print(f"False positive: {self.false_positive}")
        print(f"False negative: {self.false_negative}")
        print(f"F1 Score: {self.f1_score}")

# Take in the image directory, return a dictionary of {"image_name": cv_object} pair
def readAllImages(img_dir):
    images = {}
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            if f.endswith('.png') or f.endswith('.jpg'):
                img_path = os.path.join(img_dir, f)
                img = cv2.imread(img_path)
                # OpenCV opens image as BGR, covert it back to RGB format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[f] = img
        return images
    else:
        print("ERROR: " + img_dir + " doesn't exist")
        return None

# Run the machine learning classifier
# Input: images - a dictionary of {"image_name": cv_object} pair}
#        expected_class - the expected class of the images
#        stats - a dictionary of true/false positive/negative with their counts
# Return: total number of images ran
# NOTE: This function will modify stats
def ml_classification(images, expected_class, stats):
    model_path = "ML_Classifier/First_Test.h5"
    model = keras.models.load_model(model_path)
    model_key = ["mask_weared_incorrect", "with_mask", "without_mask"]

    # Number of images
    num_images = 0
    for key in images:
        img = images[key]
        # Count the total number of images ran
        num_images = num_images + 1
        tmp = tf.image.resize(img, (200, 200)) # Model takes 200x200 image
        tmp = tf.keras.preprocessing.image.img_to_array(tmp, data_format=None, dtype=None) / 255
        tmp = np.expand_dims(tmp, axis=0)
        prediction = model.predict(tmp)
        if expected_class == "with_mask":
            if model_key[np.argmax(prediction[0])] == expected_class:
                stats["true_positive"] = stats["true_positive"] + 1
            else:
                stats["false_negative"] = stats["false_negative"] + 1

        elif expected_class == "without_mask":
            if model_key[np.argmax(prediction[0])] == expected_class:
                stats["true_negative"] = stats["true_negative"] + 1
            else:
                stats["false_positive"] = stats["false_positive"] + 1
    return num_images

# Run the edge detection method
# Input: images - a dictionary of {"image_name": cv_object} pair}
#        expected_class - the expected class of the images
#        stats - a dictionary of true/false positive/negative with their counts
# Return: total number of images ran
# NOTE: This function will modify stats
def edge_detection(images, expected_class, stats):
    model_path = "Edge_Detection/Canny_First_Test.h5"
    model = keras.models.load_model(model_path)
    model_key = ["mask_weared_incorrect", "without_mask", "with_mask"]

    # Number of images
    num_images = 0
    for key in images:
        img = images[key]
        # Count number of images ran
        num_images = num_images + 1 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Canny edge detection
        canny = cv2.Canny(gray, 100,200)
        canny = cv2.resize(canny,(200,200))
        vector = canny.flatten()
        vector = np.expand_dims(vector, axis=0)
        prediction = model.predict(vector)
        if expected_class == "with_mask":
            if model_key[np.argmax(prediction[0])] == expected_class:
                stats["true_positive"] = stats["true_positive"] + 1
            else:
                stats["false_negative"] = stats["false_negative"] + 1

        elif expected_class == "without_mask":
            if model_key[np.argmax(prediction[0])] == expected_class:
                stats["true_negative"] = stats["true_negative"] + 1
            else:
                stats["false_positive"] = stats["false_positive"] + 1
    return num_images

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

# Run the facial landmark method
# Input: images - a dictionary of {"image_name": cv_object} pair}
#        expected_class - the expected class of the images
#        stats - a dictionary of true/false positive/negative with their counts
# Return: total number of images ran
# NOTE: This function will modify stats
def facial_landmark(images, expected_class, stats):
    shape_predictor = "Feature_Recognition/feature_constellations.dat"
    # initialize dlib's feature detector (HOG + constellation ) and then create
    # the facial landmark predictor
    feature_detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(shape_predictor)
    num_images = 0
    
    for key in images:
        num_images = num_images + 1
        img = images[key]
        gauss = gaussian_filter(3, sigma=5, m=0)
        k=3
        edge = np.diag(np.zeros(k-1)+-1,-1) + np.diag(np.zeros(k-1)+-1,1)
        edge[1][1] = 6
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply filters
        edgd = cv2.filter2D(src=gray, ddepth=-1, kernel=edge)    # accentuate edges
        filtd = cv2.filter2D(src=edgd, ddepth=-1, kernel=gauss)  # smooth out low delta sections
        # detect faces in the filtered grayscale image
        rects,scores,idx = feature_detector.run(filtd, 2, .25)
        # if no feature constellations are found, mask is present
        if ( len(rects) < 1 ):
            if expected_class == "with_mask":
                stats["true_positive"] = stats["true_positive"] + 1
            elif expected_class == "without_mask":
                stats["false_positive"] = stats["false_positive"] + 1
        else: # No mask is present
            if expected_class == "with_mask":
                stats["false_negative"] = stats["false_negative"] + 1
            elif expected_class == "without_mask":
                stats["true_negative"] = stats["true_negative"] + 1
    return num_images

def runFaceDetectionCam():
    # For ML classification
    ml_classification_model_path = "ML_Classifier/First_Test.h5"
    ml_classification_model = keras.models.load_model(ml_classification_model_path)
    ml_classification_model_key = ["mask_weared_incorrect", "with_mask", "without_mask"]

    # For edge detection
    edge_detection_model_path = "Edge_Detection/Canny_First_Test.h5"
    edge_detection_model = keras.models.load_model(edge_detection_model_path)
    edge_detection_model_key = ["mask_weared_incorrect", "without_mask", "with_mask"]

    # For facial landmark
    feature_detector = dlib.get_frontal_face_detector()
    shape_predictor="Feature_Recognition/feature_constellations.dat"      #all features
    predictor = dlib.shape_predictor(shape_predictor)

    # define a video capture object
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    detection_method = "Edge Detection"
    while(True):
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        x, y, w, h = 0, 0 ,0 ,0
        prediction_key = "None"
        if detection_method == "Edge Detection":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Canny edge detection
            canny = cv2.Canny(gray, 100,200)
            canny = cv2.resize(canny,(200,200))
            vector = canny.flatten()
            vector = np.expand_dims(vector, axis=0)
            prediction = edge_detection_model.predict(vector)
            prediction_key = edge_detection_model_key[np.argmax(prediction[0])]
        elif detection_method == "ML Classification":
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tmp = tf.image.resize(tmp, (200, 200)) # Model takes 200x200 image
            tmp = tf.keras.preprocessing.image.img_to_array(tmp, data_format=None, dtype=None) / 255
            tmp = np.expand_dims(tmp, axis=0)
            prediction = ml_classification_model.predict(tmp)
            prediction_key = ml_classification_model_key[np.argmax(prediction[0])]
        elif detection_method == "Facial Landmark":
            gauss = gaussian_filter(3, sigma=5, m=0)
            k=3
            edge = np.diag(np.zeros(k-1)+-1,-1) + np.diag(np.zeros(k-1)+-1,1)
            edge[1][1] = 6
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # apply filters
            edgd = cv2.filter2D(src=gray, ddepth=-1, kernel=edge)    # accentuate edges
            filtd = cv2.filter2D(src=edgd, ddepth=-1, kernel=gauss)  # smooth out low delta sections
            # detect faces in the filtered grayscale image
            rects,scores,idx = feature_detector.run(filtd, 2, .25)
            # if no feature constellations are found, mask is present
            if ( len(rects) < 1 ):
                prediction_key = "with_mask"
            else:
                prediction_key = "without_mask"
                for i, rect in enumerate(rects):
                    # convert feature constellation into x,y coords
                    shape = predictor(frame, rect)
                    shape = to_nparray(shape)
                    # highlight exposed parts
                    for (x, y) in shape[48:68]:                         # draw mouth
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    for (x, y) in shape[27:35]:                         # draw nose
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        else:
            print("Invalid Option")
            break

        color = (0, 0 ,0)
        # Group the mask_weared_incorrect category as with mask
        if prediction_key == "mask_weared_incorrect":
            prediction_key = "with_mask"

        if prediction_key == "with_mask":
            color = (0, 255, 0) # Green
        else: 
            color = (0, 0, 255) # Red
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, prediction_key, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, detection_method, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Switch detection method to edge detection
        if cv2.waitKey(1) & 0xFF == ord('1'):
            detection_method = "Edge Detection"

        # Switch detection method to ML classification
        if cv2.waitKey(1) & 0xFF == ord('2'):
            detection_method = "ML Classification"

        # Switch detection method to Facial Landmark
        if cv2.waitKey(1) & 0xFF == ord('3'):
            detection_method = "Facial Landmark"

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

# Set this to True  if running on Google Colab
USE_GOOGLE_COLAB = False

if USE_GOOGLE_COLAB == False:
    def main(argv):
        if(len(argv) == 1 and argv[0] == "1"):
            # Run different methods with webcam feed
            runFaceDetectionCam()
        elif(len(argv) == 2 and argv[0] == "0"):
            # Run different methods with dataset
            img_dir = argv[1]
            # Init objects
            ml_method = faceMaskDetectionMethod("ML Classification", ml_classification, img_dir)
            edge_detection_method = faceMaskDetectionMethod("Edge Detection", edge_detection, img_dir)
            facial_landmark_method = faceMaskDetectionMethod("Facial Landmark", facial_landmark, img_dir)
            # Run 
            ml_method.run()
            edge_detection_method.run()
            facial_landmark_method.run()
            # Print results
            ml_method.printResult()
            edge_detection_method.printResult()
            facial_landmark_method.printResult()
        else:
            print("Usage (run detection on webcam): python3 s_14.py 1")
            print("OR")
            print("Usage (run detection on dataset): python3 s_14.py 0 <img folder>")
            print("The img folder has to contain two folders, one is called \"without_mask\" and one is called \"with_mask\", which contain images for each class accordingly")
            exit(1)
    if __name__ == "__main__":
        main(sys.argv[1:])
else:
    img_dir = 'Dataset/'
    # Init objects
    ml_method = faceMaskDetectionMethod("ML Classification", ml_classification, img_dir)
    edge_detection_method = faceMaskDetectionMethod("Edge Detection", edge_detection, img_dir)
    facial_landmark_method = faceMaskDetectionMethod("Facial Landmark", facial_landmark, img_dir)
    # Run 
    # NOTE: Runtime profiling can easily be done using %time in Google Colab
    ml_method.run()
    edge_detection_method.run()
    facial_landmark_method.run()
    # Print results
    ml_method.printResult()
    edge_detection_method.printResult()
<<<<<<< HEAD
    facial_landmark_method.printResult()
=======
    facial_landmark_method.printResult()
>>>>>>> 11fd166 (Adding Files)
