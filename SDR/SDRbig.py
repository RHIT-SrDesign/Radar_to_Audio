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
TIMETOSHOW = 1e-6
STARTFREQ = 100e6
ENDFREQ = 500e6

# SDR Parameters
sample_rate = 5e6 # samples per second
center_freq = STARTFREQ # define a center frequency of 100 MHz for sampling
num_samples = 256 # number of data points per call to rx()

numrows = 1 # get at least this many rows of data before advancing the center frequency

# SDR max frequency points, equal to the sample rate with twice niquiust rate
FREQPOINTS = num_samples 
bw = sample_rate

# find the number of sweeps
numsweeps = math.ceil((ENDFREQ-STARTFREQ)/bw)

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
    xticks = np.arange(num_samples,step=num_samples/10)

    # axis labels
    xlabels = center_freq+np.arange(-bw/2,bw/2,step=bw/10)
    xlabels = xlabels/1e6
    print(xlabels)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
 
    # label the plot axis
    plt.xlabel("MHz")
    plt.ylabel("Magnitude")

    return fig, ax

def fixAxis(ax):
    # axis ticks
    xticks = np.arange(num_samples,step=num_samples/10)

    # axis labels
    xlabels = center_freq+np.arange(-bw/2,bw/2,step=bw/10)
    xlabels = xlabels/1e6
    print(xlabels)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)


def plot(fig1, ax,data,rows):
    print("show new plot")

    # start timer for timing plot function
    start = datetime.now()

    global fignum
    global figs

    # find the hitmap array to plot
    hitmap_array_db = fft_intensity_plot(data, rows, FREQPOINTS, 2, 300)
    #max_points_db = max_points_plot(data,rows,FREQPOINTS,2,300)

    # scale the columns of hitmap_array_db to be 1024 wide to fit the screen well

    # show the image of the hitmap
    cmap = 'plasma'
    ax.imshow(hitmap_array_db, origin='lower', cmap=cmap, interpolation='bilinear')
    ax.set_aspect('auto')
    
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

def max_points_plot(data: np.ndarray, num_ffts,fft_len: int = 256, fft_div: int = 2, mag_steps: int = 100):
    
    max_points = np.zeros([1,fft_len])

    for i in range(fft_len):
        for m in range(num_ffts):
            if(data[m][i] > max_points[i]):
                max_points[i] = data[m][i]

    max_points_db = 20.0 * np.log10(max_points+1)

    return max_points_db

def RollAndReplace(A, B, lenB):
    # quickly roll A and tack rows of B under A
    if(lenB < np.size(A,axis=0)):
        rowsToDelete = np.arange(lenB)
        A = np.delete(A, rowsToDelete, axis=0)
        A = np.vstack((A, B))
    else:
        A = B
    return A

def ScrollAndReplace(A, B, widB):
    # quickly scroll A and tack columns of B on the right of A
    if(widB<np.size(A,axis=1)):
        colsToDelete=np.arange(widB)
        A = np.delete(A,colsToDelete,axis=1)
        A=np.hstack((A,B))    
    else:
        A=B
    return A



def main():
    # init SDR
    sdr = initSDR()

    global fignum
    global center_freq
    global bw

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
        

        # define data array to save
        #fullDataRows = int(sample_rate * timeToCapture)
        #fullData = np.empty([fullDataRows,FREQPOINTS])

        # define data array to plot
        plotDataRows = numrows
        plotData = np.empty([plotDataRows,numsweeps*FREQPOINTS])

        # store newDataWindow and newDataWindowSize
        newDataWindowMaxRows = numrows
        newDataWindow = np.empty([newDataWindowMaxRows,FREQPOINTS])
        
        # store how many new data points are in the window, start with zero
        newDataWindowSize = 0 

        # while we have frequency left to sweep
        while center_freq<ENDFREQ:
            
            # wait for new data

            # perhaps thread the newData PSD finding, the plotting, and the data rolling into different threads

            # get data from SDR, and the number of rows in new data points
            newData = getData(sdr)
            newDataSize = num_samples

            # make a reshaped psd with the data
            newData = shapePSD(newData,newDataSize)        

            # correct for the new size
            newDataSize = np.size(newData,axis=0)
            #dataReady = false

            # append newData rows to fullData rows
            #fullData = np.append(fullData,newData,axis=0)

            newDataWindow = RollAndReplace(newDataWindow,newData,newDataSize)
            # keep track of how many rows the new data is
            newDataWindowSize = newDataWindowSize + newDataSize
            
            # capture numrows at once
            if(newDataWindowSize > numrows):

                # slide the window for plotData (fix)
                plotData = ScrollAndReplace(plotData,newDataWindow,FREQPOINTS)

                # plot data onto window
                print("refresh")
                
                print(np.shape(plotData))
                fignum = fignum + 1
                plot(fig1,ax, plotData,numrows)

                # advance the center frequency
                center_freq = center_freq + bw
                sdr.rx_lo = int(center_freq) 

                # redraw the axis
                fixAxis(ax)

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
