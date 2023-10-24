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

# define the size of the window
UISIZE = np.array([10,6])

# show 1 uS of data at once
TIMETOSHOW = 1e-3

# SDR Parameters
sample_rate = 1e6 # samples per second
center_freq = 100.5e6 # define a center frequency of 100 MHz for sampling
num_samples = 1024 # number of data points per call to rx()

numrows = 100 # get at least this many rows of data before refreshing the plot

# SDR max frequency points, equal to the sample rate with twice niquiust rate
FREQPOINTS = num_samples 
bw = sample_rate

dataReady=True

# plot params
window = 'hamming' # Type of window
nperseg = 180 # Sample per segment
noverlap = int(nperseg * 0.7) # Overlapping samples
nfft = 256 # Padding length
return_onesided = False # Negative + Positive 
scaling = 'spectrum' # Amplitude

freq_low, freq_high = 600, 1780
time_low, time_high = 0.103, 0.1145

figs = []
fignum = 0

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


def getData(sdr):
    # return a 1D array of data from the SDR not yet seen yet
    data = sdr.rx()
    return data


def makeWindow(size):
    # Create the window
    fig, ax = plt.subplots(figsize=(UISIZE[0],UISIZE[1]))
    # return the window for use by the rest of the program

    # axis ticks
    bwtenthMHz = bw / 1e5
    xlabels = (bwtenthMHz/2)-np.arange(bwtenthMHz)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    print(xlabels)
    # label the plot axis
    plt.xlabel("MHz")
    plt.ylabel("Magnitude")

    return fig, ax


def plot(fig1, ax,data,rows):
    print("show new plot")

    # start timer for timing plot function
    start = datetime.now()

    global fignum
    global figs

    # find the hitmap array to plot
    hitmap_array_db = fft_intensity_plot(data, rows, FREQPOINTS, 2, 300)
    
    # show the image of the hitmap
    cmap = 'plasma'
    ax.imshow(hitmap_array_db, origin='lower', cmap=cmap, interpolation='bilinear')
    
    # append the fig to the figure array
    figs.append(fig1)

    # draw a new figure
    fig1.canvas.draw()
    plt.pause(0.1)
    
    # stop timer for timing plot function
    stop = datetime.now()
    timeelapse = stop-start
    print("plotted data in ",end='')
    print(timeelapse)

def saveFigs():
    # start timer for timing plot function
    start = datetime.now()
    
    global figs
    global fignum

    # save all the figs in the array
    for i in range(fignum):
        filename = f"fig_intensity{i}.png"
        figs[i].savefig(filename, dpi=300, format='png')

    # stop timer for timing plot function
    stop = datetime.now()
    timeelapse = stop-start
    print("saved figs in ",end='')
    print(timeelapse)



def shapePSD(data,num_ffts):
    # Calculate power spectral density (frequency domain version of signal)
    psd = np.abs(np.fft.fftshift(np.fft.fft(data)))**2
    psd_dB = 10*np.log10(psd)

    # normalize fft  
    max_mag = np.amax(psd_dB)
    #print(max_mag)
    min_mag = np.abs(np.amin(psd_dB))

    for i in range(num_ffts):
        psd_dB[i] = (psd_dB[i]+(min_mag))/(max_mag+(min_mag)) 

    # reshape fft into 1 by num_ffts matrix
    psd_dB = psd_dB.reshape(1,-1)    

    return psd_dB

def fft_intensity_plot(data: np.ndarray, num_ffts,fft_len: int = 256, fft_div: int = 2, mag_steps: int = 100):
    # credit for parts of fft_intensity_plot https://teaandtechtime.com/python-intensity-graded-fft-plots/
    mag_step = 1/mag_steps

    hitmap_array = np.random.random((mag_steps+1,int(fft_len/fft_div)))*np.exp(-10)

    for i in range(num_ffts):
        for m in range(fft_len):
            hit_mag = int(data[i][m]/mag_step)
            hitmap_array[hit_mag][int(m/fft_div)] = hitmap_array[hit_mag][int(m/fft_div)] + 1

    hitmap_array_db = 20.0 * np.log10(hitmap_array+1)
    
    return(hitmap_array_db)

def RollAndReplace(A, B, lenB):
    # quickly roll A and tack B onto A
    if(lenB < np.size(A,axis=0)):
        rowsToDelete = np.arange(lenB)
        A = np.delete(A, rowsToDelete, axis=0)
        A = np.vstack((A, B))
    else:
        A = B
    return A




def main():
    # init SDR
    sdr = initSDR()

    global fignum

    #sdr = 1
    #threadname=threading.Thread(target=)

    #threadname.daemon = True

    #threadname.start()
    # capture real time or capture and save?
    #capturemode = input("real time capture [RT] or capture and save [CS]? \n")
    capturemode = "CS"

    if capturemode == "CS":

        # create live capture window
        fig1, ax = makeWindow(UISIZE)

        # get filename, bandwidth, time to capture, s/s, refresh rate
        #filename = input("input file name to save as \n")
        filename = "savedata.csv"
        #timeToCapture = input("input time to capture in seconds \n")
        timeToCapture = 60
        #refreshRate = input("input refresh rate in seconds \n")
        refreshRate = 1

        # define data array to save
        fullDataRows = int(sample_rate * timeToCapture)
        #fullData = np.empty([fullDataRows,FREQPOINTS])
        # define data array to plot
        plotDataRows = int(sample_rate * TIMETOSHOW)
        plotData = np.empty([plotDataRows,FREQPOINTS])

        #define some times
        startTime = time.time()
        runTime = startTime+timeToCapture
        lastRefresh = startTime


        # store newDataWindow and newDataWindowSize
        # calculate how big it needs to be, add 20% just in case
        newDataWindowMaxRows = min(math.ceil(sample_rate * refreshRate * 1.2),plotDataRows)
        
        newDataWindow = np.empty([newDataWindowMaxRows,FREQPOINTS])
        # store how many new data points are in the window, start with zero
        newDataWindowSize = 0 

        # while time < time to capture
        while time.time() < runTime:
            
            # wait for new data
            while dataReady == False:
                # do nothing
                pass
            
            # perhaps thread the newData PSD finding, the plotting, and the data rolling into different threads

            # get data from SDR, and the number of rows in new data points
            newData = getData(sdr)
            newDataSize = num_samples

            # make a reshaped psd with the data
            newData = shapePSD(newData,newDataSize)        

            newDataSize = np.size(newData,axis=0)
            #dataReady = false

            # append newData rows to fullData rows
            #fullData = np.append(fullData,newData,axis=0)

            # if we have room in our buffer
            if (newDataWindowSize+newDataSize) <= plotDataRows:
                # append newData to newdatawindow
                newDataWindow = RollAndReplace(newDataWindow,newData,newDataSize)
                # keep track of how many rows the new data is
                newDataWindowSize = newDataWindowSize + newDataSize
            else:
                print("WARNING: Overwriting plot data, plot refresh rate or plot buffer too small")

                newDataWindow = RollAndReplace(newDataWindow,newData,newDataSize)

                newDataWindowSize = plotDataRows

            # if its time to refresh the plot
            #if (time.time() > lastRefresh + refreshRate):
            if(newDataWindowSize > numrows):
                # reset lastRefresh to current time
                lastRefresh = time.time()

                # get rid of all non-data in the newDataWindow
                print(newDataWindowSize)
                rowsToDelete = np.arange(newDataWindowMaxRows - newDataWindowSize)
                newDataWindow = np.delete(newDataWindow, rowsToDelete, axis=0)

                # slide the window for plotData
                plotData = RollAndReplace(plotData,newDataWindow,newDataWindowSize)

                # plot data onto window
                print("refresh")
                
                print(np.shape(plotData))
                fignum = fignum + 1
                plot(fig1,ax, plotData,plotDataRows)

                # clear the newDataWindow
                newDataWindow = np.empty([newDataWindowMaxRows,FREQPOINTS])
                newDataWindowSize=0
            
              
    
    print("stop")
    #plt.show()
    df=pd.DataFrame(plotData)
    df.to_csv(filename)
    #saveFigs()
    print("done")
    # if capturemode = realtime
        # get data from SDR
        # store data into buffer

    
if __name__ == main():
    main()
