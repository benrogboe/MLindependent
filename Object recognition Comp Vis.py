
#Implementations of Viola Jones, Eigenface, and CNN object recognitionusing the OpenCV library.
________________________________________________________________________________
 #Viola Jones/ Haar Cascades: Works with both still images and live video.
 #This method employs hand coded features(relative locations of various image
 #elements encoded as lines or edges-vectors) which are fed into a SVM classifier.
 #Due to the fact that relative positions of image components were hand coded,
 #this algorithm lacked generalizability as objects could only be detected accurately
 #if they appeared in with the same orientation/ angle that the objects appeared in the
 #images of the hand coded dataset. Training is slow but detection is very fast.
 #Features are evaluated with integral images. This is essentially the process of
 # computing a value at each pixel that is the sum of the pixel values above and to
 #the left of it. This allows a relatively fast computation in one pass through the
 #image/ frame. Boosting is often used for feature selection to eliminate the
 #evaluation of the entire feature set. Boosting essentially combines weak learners
 #into a comparatively more accurate ensemble classifier. Training consists of multiple
 #boosting rounds in which a weak learner is selected which does well on examples that
 #were difficult for previous weak learners(hardness is captured by the weights which
 #are attached to the training examples). Initially each training example is weighed
 #equally and in each boosting round a weaker learner is found which achieves the
 #lowest weighted training error. The weights are then raised of training examples
 #which are misclassified by a current weak learner. A final classifier is computed
 #as a linear combination of all the weak learners where the weight of each learner
 #is directly proportional to the learners accuracy. Although this does not lead to
 #the fastest detection among the algorithms analyzed here it is enables very fast
 #testing, enables flexibility in the choice of weak learners, and integrates classifier
 #training with feature selection. As opposed to an SVM, used in the Eigenface algorithm,
 #training is slow and, many examples are needed, and doesn't work as well for multiclass
 #problems. In addition to this, an attentional cascade is used for fast rejection of non- face windows.
 #With this, initially simple classifiers reject many of the negative sub windows while detecting almost
 #all positive sub windows. A positive response from the first classifier triggers the evaluation of a
 #second more complex classifier and a negative outcome at any point leads to the rejection of the sub window.
 #The cascade is trained by first setting the target detection and false positive rates for each stage and features
 #are continuously added until target values have been met. If the overall false positive rate is not low enough
 #then another stage is added,and false positives from the current stage are used as negative training examples
 #for the following stage.

#Viola Jones/Haar cascades, uses webcam for facial detection. Used library functions(did not train)
#Using the OpenCV trainers(haarcascade_frontalface_default.xml, and haarcascade_eye.xml)
import cv2
import numpy as np
import os

def main():

#WebCam capture
    capWebcam = cv2.VideoCapture(0)

    #OpenCV Classifiers
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #defining openCV classifiers
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # start the webcam
    if capWebcam.isOpened() == False:
        print "webcam unavaliable \n\n"          # Error: webcam is unavailable
        os.system("pause")                                          #Pause until the uses press a Key
        return

    while cv2.waitKey(1) != 27 and capWebcam.isOpened():
        blnFrameReadSuccessfully, imgOriginal = capWebcam.read()    # Read the next frame

        if not blnFrameReadSuccessfully or imgOriginal is None:     #  Error: We can't read the next frame
            print "Cant read the next frame\n"
            os.system("pause")
            break


        # Process
        imgGrayscale = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)    # Convert original image into grayscale

        faces = face_cascade.detectMultiScale(imgGrayscale, 1.3, 5)     # apply Face classifier in the grayscale image

        for (x,y,w,h) in faces:                                         # Draw the rectangle in the face area
             cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)     # RGB: Green bounding box
             # Once we detect the face, we get a Region of Interest. looking for eyes inside the Region
             regionofinterest_imgGrayscale = imgGrayscale[y:y+h, x:x+w]
             regionofinterest_color = imgOriginal[y:y+h, x:x+w]
             eyes = eye_cascade.detectMultiScale(regionofinterest_imgGrayscale) # apply the eye classifier inside the regionofinterest_grayscale image
             for (ex,ey,ew,eh) in eyes:             #CMY: Use Cyan rectangle for bounding box for eyes
                 cv2.rectangle(regionofinterest_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)

        cv2.imshow("face detector", imgOriginal)


    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
________________________________________________________________________________
#Boosting algorithm(Viola Jones 2)
#below is pseudocode for how a boosting algorithm  would work in this context

for i =1: boost  #classes are 1 and -1
    Classifier[i] = train(X,Y,weights): #train a weighted classifier
    yhat = predict(Classifier[i], X): #get predictions
    errors = weights*(y~=yhat): #calculated weighted error rate
    alpha(i) = .5 log (1-error)/error: #calculate coefficient. when e is >.5 it means that
    #the classifier is no better than random guess
    weights += exp(-alpha(i)*y*yhat): #update the weights. y*yhat > 0 if yhat = y and the
    #weights also decrease
    weights =weights/sum(weights):
    end:
    (sum alpha(i)*predict(CLassifier[i],Xtest) > 0 #final classifier
________________________________________________________________________________
#Eigenfaces: Works in a holistic manner rather than a feature based one(such as
#in Viola-Jones). Subclass of PCA(principal component analysis-which essentially
#converts a set of possibly correlated variables into a et of variables which are
#linearly correlated through the use of an orthogonal transformation). This algorithm
#performs a PCA on grayscale bitmap on images of faces. An image is “unfolded”into
#a vector for every instance in a dataset. The resultant matrix(for the dataset) has
#the mean subtracted from it(essentially the vector representation of the average face).
#The covariance matrix is then computed and a given number of eigenvalues and eigenvectors
#which span the most variance are selected. The matrix of these eigenvectors has the same
#dimensionality as that of a given bitmapped face. These eigenvectors are then unfolded
#into a bitmap. This produces the most prominent deviations from the mean from a given dataset.
#This then is repeated for each input in a dataset. This captures variations in angle/ features
#and allow for more generalizability than viola jones/Haar cascades. A given face is represented
#through a dot produect of the original image and one of the eigenvectors. By taking a small number of
#eigenvectors a good approximation of a particular face can be done. This has much better
 #compression than those which use Haar-cascades. This can be used to detect the degree of
 #similarity between faces and can therefore be used for detection. If the “distance” between an
 #input weight weight vector and all the weight vectors is less than a threshold then a person is recognized.
 #Some limitations of this method is that it is not robust to misalignment or background variation, and
 # that the PCA assumes that the data has a gaussian distribution, so the shape of the dataset is
 #not always well discribed by its principal components. In addition to this, the direction of maximum variance is not alwasy good
 #for classification.
#EIGENFACES USING SVM(must run the different programs as seperate files)

import cv2
import NameFind

#Harrcascades for detection of features
eye = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
face =cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

recognise = cv2.face.createEigenFaceRecognizer(15,4000) #making the Eigenface recognizer
recognise.load('Recogniser/trainingDataEigan.xml') #training data loading in

cap = cv2.VideoCApture(0) #video object
ID =0
while true:
    ret, img = cap.read( ) #read the camera object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converts the camera image to grayscale
    faces = face.detectMultiScale(gray, 1.3, 5) #detects faces and stores their positions
    for(x,y,w,h) in faces: #frame locations, width and height. Confirms that the eyes are within the face
        gray_face =cv2.resize((gray[y: y+h, x: x+w]), (110,110)) #crops face
        eyes = eye.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            ID, conf = recognise.predict(gray_face) #determines ID and confidence level
            NAME = NameFind.ID2Name(ID, conf) #finds name from lookup table
            Namefind.DispID(x,y,w,h,NAME,gray)
    cv2.imshow('Eigenface Face REcognition System', gray) #displays the video
    if cv2.waitkey(1) & 0xFF == ord('q'): #quits
        break
cap.release()
cv2.destroyAllWindows()
________________________________________________________________________________
#(EIGENFACES 2)CAPTURING/ STORING PHOTOS TO GENERATE TRAINING DATA
#NOT working properly so dataset from the internet can be used

#importing Haar cascades
eye = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
face =cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

ID =NameFind.AddName() #saves name to a test file
Count = 0 #counts photos
cap = cv2.VideoCapture(0)
while Count < 50:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if np.average(gray) > 120: #verifies the avg pixel values in image above certain threshold to proceed
        faces = face_cascade.detectMultscale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            FaceImage = gray[y - int(h/2): y + int(h*1.5), x -int(x/2): x + int(w*1.5)]
            Img = (NameFind.DetectEyes(FaceImage))) # finds location of eyes and if at an angle the old face is rotated to correct the offset
            cv2.putText(gray, "Face is detected", (x+(w/2), y-5), cv2.FONT_HERSEY_DUPLEX_SMALL, .4, WHITe)
            if Img is not None:
                frame = Img
            else:
                frame =gray[y: y+h, x: x+w]
            cv2.imwrite("dataSet/User". + str(ID)) #If above conditions are satisfied the image with an ID is saved
            cv2.waitkey(300)
            cv2.imshow("CAPTURED PHOTO", frame)
            Count = Count + 1
    cv2.imshow("Face Recog system capture faces", gray)
    if cv2.waitkey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
________________________________________________________________________________
#EIGENFACES 3) #Trainer for algorithm
import os
import cv2
import numpy as np

Eigenface = cv2.face.createEigenfaceRecognizer(15) #creating the eigenface recognizer

path = 'dataset'
def getImages (path):
    ImagePaths = [os.path.join(path,f) for i in os.listdir(path)]
    FaceList = []
    IDs =[]
    for ImagePath in ImagePaths: #Imports images from dataset and saves them into a numpy array and then into a list... IDs are stored in anpther list
        faceImage = faceImage.resize((110,110)) # resizes images so EIgenface recognizer can use
        faceImage = Image.open(imagePath).convert('L') # opens a single image and converts it to grayscale
        faceNP = np.array(faceImage,'unit8') #changes the image into numpy array
        ID = int(os.path.split(imagePath)[-1].split('.')[1]) #ID of the array
        IDs.append(ID) # adds ID to the list of IDS
        FaceList.append(faceNP) #adds the array to the list
        cv2.imshow('Trainingset', faceNP) #shows all the images in the list
        cv2.waitkey(1)
    return np.array(IDs), FaceListm#Ids converted into numpy array
IDs, FaceList = getImageWithID(path)
#recognizer is trained with the ID list and image list
print('training in process')
EigenFace.train(Facelist, IDs)
print('recognizer done')
EigenFace.save('Recognizer/trainingDataEigan.xml') #saves the files
________________________________________________________________________________
#Convolutional neural nets: mainly used for classification but also good at detection.
#CNN classifiers -which were trained on extremely large datasets, scan an image and classify objects and
#objects that are detected with a high degree of certainty are included in the final visual output. Algorithms such as R-CNN reduced the number of operations,
#before running an image through a CNN, a selective search was performed which groups pixels by certain properties (texture /color). This limited number of
#image proposals are run through a CNN to compute the features for the drawn bounding box, and then an SVM classifies the image in the box. A given
#Box is then run through a linear regression model in order to create tighter boundaries.
#Other more efficient algorithms which also use large pre-trained CNNs vary from these considerably such as the YOLO object detection method.
# These algorithms divide an image into a grid and cells. This implementation doesn’t use sliding windows(typical method of creating bounding boxes).
#Applies image classification(softmax unit outputs a predicted class) and localizes for each grid element. Each grid cell is an nxnx(5+c) dimensional
# vector (specifies if there is an image in the cell and coordinates to specify the bounding box, and then a number of  classes to choose from. The target
#output is nxnx(5+c) -where n is the number of grid cell divisions and c is the number of classes, 5 denotes the binary object detection feature and the
#bounding box is specified by 4 coordinates. To train the network the input is the size of the image and it will go through a maxpool CNN(down samples and
# image representations and reduces the dimensionality of the image, and the computational cost. Done by applying a maxfilter to non overlapping subregions
#of the image) in order to map to the target output. Input x is mapped to output y with backpropagation.
#When an object spans multiple grid cells, it is assigned to a single grid cell by the midpoint of the object. A squared error loss function over all the
#elements  is commonly  used for training the network when there is a detected object, and if not the squared error for the first component only(object or no object)
#is computed. Output  predictions are run through a non-max output suppression which discards bounding boxes of low probability predictions and runs non max suppression
#for each class for the objects that were predicted to come from each class(non max suppression ensures that each object is detected only once by only taking
#the bounding box with the highest associated object detection probability and gets rid of those which overlap the box .
#Due to the fact that this is a convolutional implementation the algorithm is not implemented for each cell and is therefore faster than the other algorithms and better
#suited to real time object detection.



#CNN OBJECT DETECTION (implemeting real time algorithm from darkflow. Using their dataset and pretrained CNN)
import cv2
from darkflow.net.build import TFNEt #importing pre trained CNN from YOLO9000 creator
import numpy as np
import time

options = { #creating options dictionary. Threshold will determine the number of boxes
'model': 'cfg/yolo.cfg', # or 'cfg/tiny-yolo-voc-fs.cfg'
'load': 'bin/yolo.weights',
'threshold': 0.2,
'gpu': 0.8
}

tfnet = TFNet(options) #creating TFNet opbject + passing options
colors = [tuple(240 * np.random.rand(3)) for  _ in range(10)] #random colors (10 of them) for the bounding boxes that will be generated
capture = cv2.videoCapture(0)#capture object


while True:
    starttime = time.time() #how long each frame takes
    ret, frame = capture.read #
    results = tfnet.return_predict(frame) #make prediction
    if ret:  #if capture device is still recording will continue to make predictions
        for color, result in zip( colors, results): # 1 color per result, looping over predictions
            #pull out top left and bottom right coordinates and add the confidence interval
            label = result['label']
            font = '{ }:{ :.0f}%'.format(label, confidence*100)#display
            confidence = result['confidence']
            top_left = (result['topleft']['x'], result['topleft']['y'])
            bottom_right = (result['bottomright']['x'], result['bottomright']['y'])
            frame = cv2.rectange(frame,t1,br,color 5)
            frame = cv2.putText(frame,text,t1,cv2,FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 2) #specifies look of box

            )
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break #quit/ break out of window when q is pressed
capture.release()
cv2.destroyAllWindows()

#You can also train to detect object of our choice/ new object.
#use approx. 1000 annotated images with bounding boxes
#modify cfg to get proper final layer to train on dataset + generate new weights for model
#modify by 1 class and filter
#edit darkflow text file to include new class name
#start w/ pretrained weights.
________________________________________________________________________________
#sources: https://github.com/ITCoders/Human-detection-and-Tracking/blob/master/scripts/face_recognition.py
#https://github.com/onurvarol/Eigenface
#https://www.learnopencv.com/principal-component-analysis/
#http://setosa.io/ev/principal-component-analysis/
#https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
#https://github.com/INVASIS/Viola-Jones
#https://www.learnopencv.com/eigenface-using-opencv-c-python/, https://github.com/thtrieu/darkflow, https://pjreddie.com/darknet/yolo/,https://www.kdnuggets.com/2017/10/deep-learning-object-detection-comprehensive-review.html
