import adi
import iio
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, hamming
from scipy.fftpack import fft, fftfreq, fftshift
import pandas as pd
import random
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime
import threading

# define the size of the window
UISIZE = np.array([10,6])

# define the start and end frequency of the sweep
STARTFREQ = 0.92e9
ENDFREQ = 0.98e9
capturebw = ENDFREQ - STARTFREQ
bigcenter = (ENDFREQ + STARTFREQ) / 2


# SDR Parameters
sample_rate = 5e6 # samples per second
center_freq = STARTFREQ # define a center frequency of 100 MHz for sampling
num_samples = 256 # number of data points per call to rx()

# SDR max frequency points, equal to the sample rate with twice niquiust rate
FREQPOINTS = num_samples 
bw = sample_rate

numsweeps = math.ceil((ENDFREQ-STARTFREQ)/bw)
windowWidth = numsweeps*num_samples

# axis ticks
xticks = np.arange(windowWidth,step=windowWidth/10)

# axis labels
xlabels = bigcenter+np.arange(-capturebw/2,capturebw/2,step=capturebw/10)
# label in MHz
xlabels = xlabels/1e6

def initSDR():
    # start the timer    
    start = datetime.now()
    
    # initialize the SDR
    sdr = adi.Pluto()
    sdr.gain_control_mode_chan0 = 'manual'
    sdr.rx_hardwaregain_chan0 = 70.0 # dB
    sdr.rx_lo = int(center_freq)
    sdr.sample_rate = int(sample_rate)
    sdr.rx_rf_bandwidth = int(bw)
    sdr.rx_buffer_size = num_samples
 
    # stop the timer
    stop = datetime.now()
    timeelapse = stop-start
    print("initted sdr in ",end='')
    print(timeelapse)

    return sdr

def makeWindow():
    # Create the window
    fig, ax = plt.subplots(figsize=(UISIZE[0],UISIZE[1]))

    drawAxis(ax)

    # return the window for use by the rest of the program
    return ax

def drawAxis(ax):
    # set the x ticks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    
    plt.grid()
    plt.ylim(top=100,bottom=30)
 
    # label the plot axis
    plt.xlabel("MHz")
    plt.ylabel("Magnitude")

def getData(sdr):
    # return a 1D array of data from the SDR not yet seen yet
    data = sdr.rx()

    # reshape data into a row vector
    data = data.reshape(1,-1)    

    return data

def takefft(data):
    psd = np.abs(np.fft.fftshift(np.fft.fft(data)))**2
    psd_dB = 10*np.log10(psd)

    return psd_dB

def plot(ax,data):
    # clear the old plot
    ax.cla()

    # plot the row of the data onto the plot
    ax.plot(data.ravel())

    # redraw the axis
    drawAxis(ax)

    # refresh
    plt.draw()
    plt.pause(0.1)

def main():

    global center_freq

    # init SDR
    sdr = initSDR()

    # allocate memory for incoming data and full data array
    data = np.empty((1,num_samples))
    bigdata = np.empty((numsweeps,num_samples))

    # make a window to display data with background and axis
    ax = makeWindow()

    sweepnum = 0
    while(plt.fignum_exists(1)):
        start = datetime.now()

        # initialize the SDR with startfreq
        center_freq = STARTFREQ
        sdr.rx_lo = int(center_freq)

        sweepnum = sweepnum + 1
        for i in range(numsweeps):        
            # get the data from the SDR in a [1 x num_samples] array
            data = getData(sdr)

            # take the fft of the data 
            data = takefft(data)

            # add data to bigdata
            bigdata[i] = data
            
            # plot the data onto the graph
            if plt.fignum_exists(1):
                plot(ax,bigdata)            

            # increase the center frequency
            center_freq = center_freq + bw
            sdr.rx_lo = int(center_freq)

        # save the data to a file
        filename = f"dump{sweepnum}.csv"
        df=pd.DataFrame(bigdata.ravel())
        df.to_csv(filename)

        stop = datetime.now()
        timeelapse = stop-start
        print("swept frequencies in ",end='')
        print(timeelapse)

if __name__ == main():
    main()
