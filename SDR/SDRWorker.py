import adi
import iio
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, hamming
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import butter,filtfilt
from scipy.signal import resample
import scipy
from scipy import signal
import pandas as pd
import random
import math
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime
import threading
import csv
from itertools import zip_longest
import warnings



def initSDR(RXGAIN,center_freq,sample_rate,bw,num_samples):
    # start the timer    
    start = datetime.now()
    
    # initialize the SDR
    sdr = adi.Pluto()
    
    #sdr.gain_control_mode_chan0 = "hybrid"
    #sdr.gain_control_mode_chan0 = "slow_attack"
    #sdr.gain_control_mode_chan0 = "fast_attack"
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RXGAIN # dB

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

def makeWindow(xticks,xlabels):
    # Create the window
    fig, ax = plt.subplots(figsize=(UISIZE[0],UISIZE[1]))

    # set the x ticks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    
    plt.grid()
    plt.ylim(top=50,bottom=-30)
 
    # label the plot axis
    plt.xlabel("MHz")
    plt.ylabel("Magnitude")


    # return the window for use by the rest of the program
    return ax

def getData(sdr,num_samples):
    # return a 1D array of complex128 data from the SDR not yet seen yet
    data = sdr.rx()

    # reshape data into a row vector
    data = data.reshape(1,-1)

    # remove the DC component from the data
    ave = np.sum(data)/num_samples
    data = data - ave    

    return data

def takefft(data):
    psd = np.abs(np.fft.fftshift(np.fft.fft(data)))**2
    psd_dB = 10*np.log10(psd)

    return psd_dB

def plot(ax,data,xticks,xlabels):
    # clear the old plot
    ax.cla()

    # plot the row of the data onto the plot
    ax.plot(data.ravel())

    # set the x ticks and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    
    plt.grid()
    plt.ylim(top=50,bottom=-30)
 
    # label the plot axis
    plt.xlabel("MHz")
    plt.ylabel("Magnitude")

    # refresh
    plt.draw()
    plt.pause(0.1)

def addToFile(data,capnum,captime,caplen):
    global done
    done = False

    global rows

    # write the capture number and the capture time to the file
    capnumrow = rows[4]
    captimerow = rows[5]
    caplenrow = rows[6]

    capnumrow.append(f'{capnum}')
    captimerow.append(f'{captime}')
    caplenrow.append(f'{caplen}')

    # define the shape of the data
    [numrows,numcolumns] = np.shape(data)

    # convert data from a matrix of samples by sweeps to a 1d array
    data = data.ravel()

    # reshape the data from IQIQIQIQIQIQIQIQ to [IIIIIIIIIIII;QQQQQQQQQQQQ]
    data = np.column_stack((np.real(data), np.imag(data)))

    ## data now looks like this:
    # I1 Q1
    # I1 Q1
    # I1 Q1
    # ...

    # write IQ data to file
    datarow = 0
    for rownum in range(len(rows)):
        # only write data in the data section
        if rownum > 7:
            # write I and Q
            rows[rownum].append(f'{data[datarow][0]}')
            rows[rownum].append(f'{data[datarow][1]}')
            datarow += 1   
    
    done = True

    
def remix(data,freqdif,sample_rate):
    # mix the data up or down by freqdif 
    
    # Perform FFT
    fft_result = np.fft.fft(data)
    
    # generates from -25 to 25 MHz
    # Generate frequency axis
    frequencies = np.fft.fftfreq(len(data), 1/sample_rate)

    # Apply frequency shift
    shifted_fft_result = fft_result * np.exp(1j * 2 * np.pi * freqdif * frequencies)


    # Perform IFFT with shifted frequencies
    data = np.fft.ifft(np.fft.fftshift(shifted_fft_result))

    return data


def cap(start,stop,numcaps,Filename,limplot):

    # define the size of the window
    global UISIZE 
    UISIZE = np.array([10,6])

    # done saving or not
    global done
    done = True

    # define the start and end frequency of the sweep
    STARTFREQ = start
    ENDFREQ = stop
    capturebw = ENDFREQ - STARTFREQ
    bigcenter = (ENDFREQ + STARTFREQ) / 2


    # SDR Parameters
    sample_rate = 50e6 # samples per second
    plutomax = 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

    center_freq = STARTFREQ # define a center frequency of 100 MHz for sampling
    num_samples = 1024 # number of data points per call to rx() after smoothing

    # gain parameters
    RXGAIN = 60 # 0-90 dB

    # SDR max frequency points, equal to the sample rate with twice niquiust rate
    FREQPOINTS = num_samples 
    bw = sample_rate

    numsweeps = math.ceil((ENDFREQ-STARTFREQ)/bw)
    windowWidth = numsweeps*num_samples
    if limplot:
        # axis ticks
        xticks = np.arange(windowWidth,step=windowWidth/10)

        # axis labels
        xlabels = bigcenter+np.arange(-capturebw/2,capturebw/2,step=capturebw/10)
        # label in MHz
        xlabels = xlabels/1e6
        # make a window to display data with background and axis
        ax = makeWindow(xticks,xlabels)

    # ignore dividebyzero warning 
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # init SDR
    sdr = initSDR(RXGAIN,center_freq,sample_rate,bw,num_samples)

    # init file for dump file
    filename = Filename
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        ## Write the CSV in the following format:
        # date of capture, 'date' # indicates the date and time of the following data recording
        # sample_rate,    'sample_rate' # indicates the sample rate at which the following data was taken in Hz
        # center_freq,    'center_freq' # indicates the center frequency at which the data was taken in Hz
        # span,           'freq_span'   # the frequency span that the data was taken in Hz
        # capture #,      1,            2,          3, ...   # indicates the capture number that the following column of data belongs to
        #capture time,   t1,           t2,         t3, ...   # indicates the time at which the capture began, in seconds
        #data,
        #1,              i1,    q1,    i2,   q2,   i3,   q3,  # iq data at t1 = 0
        #2,              i1,    q1,    i2,   q2,   i3,   q3,  # iq data at t2 = t1 + 1/sample_rate
        #3,              i1,    q1,    i2,   q2,   i3,   q3,  # iq data at t3 = t2 + 1/sample_rate
        # ... and so on and so forth

        
    

        header = [  
            ["date of capture",datetime.now()],
            ["sample_rate", sample_rate],
            ["center_freq", bigcenter],
            ["span",capturebw],
            ["capture #"],
            ["capture time"],
            ["capture length"],
            ["data"]
        ]
        writer.writerows(header)

        # write the first column with the data index
        for i in range(numsweeps*num_samples):
            writer.writerow([str(i)])

    with open(filename, 'r', newline='') as file:
        # list the rows for later editing
        reader = csv.reader(file)
        global rows
        rows = list(reader)

        

    # allocate memory for incoming data and full data array
    data = np.empty(shape=(1,num_samples))
    bigdata = np.empty(shape=(numsweeps,num_samples))
    
    rawdata = np.empty(shape=(1,num_samples),dtype=np.complex128)
    bigrawdata = np.empty(shape=(numsweeps,num_samples),dtype=np.complex128)

    bigstart = datetime.now()
    for k in range(numcaps):
        start = datetime.now()

        # initialize the SDR with startfreq
        center_freq = STARTFREQ
        sdr.rx_lo = int(center_freq)

        
        for i in range(numsweeps):        
            # get the data from the SDR in a [1 x num_samples] array, save it to a big array
            rawdata = getData(sdr,num_samples)
        
            if(limplot):
                # take the fft of the data 
                data = takefft(rawdata)

            # mix/demix the raw data up or down a little bit to be all in the same center frequency
            freqdif = bigcenter - center_freq 
            rawdata = remix(rawdata,freqdif,sample_rate)

            # save it to the big array
            bigrawdata[i] = rawdata

        
            
            # plot the data onto the graph
            if limplot:
                # subtract off the rx gain 
                data = data - RXGAIN

                # add data to bigdata
                bigdata[i] = data

                plot(ax,bigdata,xticks,xlabels)            

            # increase the center frequency
            center_freq = center_freq + bw
            sdr.rx_lo = int(center_freq)

        if k > 0:
            while not done:
                #print("wait")
                pass # do nothing, wait for the previous saving to be done
            
            t3.join()
            
        # get the time elapsed for the capture
        stop = datetime.now()
        timeelapse = stop-start
        timeelapse = float(timeelapse.total_seconds())
        print("swept frequencies in ",end='')
        print(timeelapse)

        # get the time started for each capture
        bigtimeelapse = start-bigstart
        bigtimeelapse = float(bigtimeelapse.total_seconds())

        # save the data to the file
        #may need to save time at which the grab was performed, or it may be fast enough
        t3 = threading.Thread(target=addToFile,args=(bigrawdata,k,bigtimeelapse,timeelapse), daemon=True)
        t3.start()

        
    
    # clean up the threads    
    t3.join()
        # write the updated rows back to the file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    
    #global done
    
    return True
