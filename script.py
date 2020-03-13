"""import serial # if you have not already done so
ser = serial.Serial('/dev/tty.usbserial', 9600)
ser.write(b'center x coordinate of face / pixel width of image * 180') # else send '255'"""
import serial
import bz2

import os
from model import create_model
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer
from urllib.request import urlopen

import pickle

from data import triplet_generator

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
import cv2

import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import f1_score, accuracy_score

import numpy as np
import os.path

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

encoder = LabelEncoder()
encoderfile = "encoder.sav"
encoder = pickle.load(open(encoderfile, 'rb'))

alignment = AlignDlib('models/landmarks.dat')
modelfile = "svm_one.sav"
svc = LinearSVC()
svc = pickle.load(open(modelfile, 'rb'))

cap = cv2.VideoCapture(-1)
loop = 5e10
while(True):
    try:
        # Capture frame-by-frame
        ret, jc_orig = cap.read()

        #jc_orig = cv2.resize(jc_orig, dim, interpolation = cv2.INTER_AREA)
        bb = alignment.getLargestFaceBoundingBox(jc_orig)##------------------------return this
        print(bb)
        #if(bb==None):
        #    continue
        #print(bb.left())
        x = (bb.left() + bb.right())//2
        y = (bb.top() + bb.bottom())//2
        #w = bb.right() - bb.left()
        #h = bb.bottom() - bb.top()

        #centre = (jc_orig.shape[0] + bb[0], jc_orig.shape[1] + bb[1])
        image= cv2.circle (jc_orig, (x,y), radius = 1, color = (0,0,255),  thickness = -1)
        jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
        ser = serial.Serial('/dev/ttyACM0', 9600)
        serial_write = x/jc_orig.shape[0]
        ser.write(serial_write.encode(ASCII))
        # Our operations on the frame come here
        jc_aligned = (jc_aligned /255.).astype(np.float32)
        embedded = nn4_small2_pretrained.predict(np.expand_dims(jc_aligned, axis=0))
        prediction = svc.predict(embedded)
        identity = encoder.inverse_transform(prediction)[0] ##-----------------------return this

        print(identity)
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
	    ser = serial.Serial('/dev/ttyACM0', 9600)
	    ser.write(b'255')
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
