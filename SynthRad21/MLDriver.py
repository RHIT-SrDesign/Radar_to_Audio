# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def runML(data):

    # load the model
    model = load_model('temp/checkpoint/modclass04_2500.h5')

    # decimate and feed to ML
    x = prepare(data)


    # do the ML stuff
    yh = model.predict(x).argmax(1)

    global PRI, PW, EmitterID
    PRI = 
    PW = 
    EmitterID = yh

    return True

def getInfo():

    return [PRI, PW, EmitterID]

def prepare(data):
    # Generate all of the features that the model will have available to train on
    dataOut = np.empty(size=[1,1024,8])

    # decimation factor of 100
    D = 100

    x_short = data[np.arange(0, int(np.size(data)/D), 1)]
    x_long = data[np.arange(0, np.size(data), D)]
    xf_short = fft(x_short)
    xf_long = fft(x_long)
    x_ms = abs(x_short)
    x_ps = np.angle(x_short)
    x_ml = abs(x_long)
    x_pl = np.angle(x_long)
    xf_ms = abs(xf_short)
    xf_ps = np.angle(xf_short)
    xf_ml = abs(xf_long)
    xf_pl = np.angle(xf_long)

    dataOut[i,:,0] = x_ms
    dataOut[i,:,1] = x_ps
    dataOut[i,:,2] = x_ml
    dataOut[i,:,3] = x_pl
    dataOut[i,:,4] = xf_ms
    dataOut[i,:,5] = xf_ps
    dataOut[i,:,6] = xf_ml
    dataOut[i,:,7] = xf_pl

    return dataOut
