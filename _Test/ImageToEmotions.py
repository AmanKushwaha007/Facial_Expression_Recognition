# import the required modules
# This is a code for showing the values of expression in 7 form of expression
# here are the all 7 form
# Neutral , Fear, Happy, Sad, Surprise, Disgusted, Angry
# You can simply watch that in console
import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# read image
img = cv2.imread('image.jpg')

def GiveImage():
    # call imshow() using plt object
    plt.imshow(img[:, :, ::-1])

    # display that image
    plt.show()

def GiveEmotionsValue():
    # storing the result
    x = time.time()
    print(x)
    result = DeepFace.analyze(img, actions=['emotion'])

    # print result
    print("Aman")
    print(result)
    print(result[0]['emotion'])
    print(result[0]['dominant_emotion'])

    y = time.time()
    print(y)


    x = time.time()
    print(y-x)

GiveEmotionsValue()