import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import json
import SDRWorker

# FUNCTION: open(filename) opens the file saved in the filename and opens the file and returns a pointer towards the raw data matrix, along with the center frequency to which it was recorded at

# FUNCTION: openLive(filename) opens the file currently being recorded in, returns a pointer towards the most recent data stream, and an index to which sweep we are in, along with the center frequency at which the data is currently being recorded

# FUNCTION: returnMeta(filename) returns the metadata of the file at filename
def returnMeta(file_path):

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)

        
        captime = 0

        try:
            captime = datetime.strptime(rows[0][1], "%Y-%m-%d %H:%M:%S")
    
        except ValueError as e:
            print(f"Error: {e}")
        # return time, sample_rate, center_freq, freq_span
        sample_rate = float(rows[1][1])
        center_freq = float(rows[2][1])
        freq_span = float(rows[3][1])

        # get the start times and the stop times and return them as row vectors
        # here is where id do it [int(s) for s in rows[4][1:]]

    return [captime,sample_rate,center_freq,freq_span] 

def normresh(data,samples,sweeps):
    # collapses the data matrix given in data along the time axis to create samples sample long representation of the signal that may have been present in that sweep. Then completes this sweeps times for each sweep. returns a 1 x (samples*sweeeps) array.

    # create an empty array to store all this data in
    normresh_data = np.empty(shape=(sweeps,samples),dtype=np.complex128)

    #print(np.shape(data))
    for i in range(1,sweeps,1):
        # grab the current column
        cur_data = data[:,i]
        # reshape this columm into a matrix of every row has samples number of elements and as many columns as we need
        reshaped_data = cur_data.reshape(-1, samples)
        # sum along the columns to get a single samples long array
        summed_subarrays = np.sum(reshaped_data, axis=0)
        
        #print(np.shape(summed_subarrays))

        # add this to our data array to store
        normresh_data[i][:] = summed_subarrays
        
    # flatten the normresh_data to get a single 1xsamples*sweeps array
    
    normresh_data = normresh_data.ravel()

    return normresh_data

def runProc(start,stop,numcaps,limplot):
    # read in CSV file
    # date of capture, 'date' # indicates the date and time of the following data recording
    # sample_rate,    'sample_rate' # indicates the sample rate at which the following data was taken in Hz
    # center_freq,    'center_freq' # indicates the center frequency at which the data was taken in Hz
    # span,           'freq_span'   # the frequency span that the data was taken in Hz
    # capture #,      1,            2,          3, ...   # indicates the capture number that the following column of data belongs to
    # capture time,   t1,           t2,         t3, ...   # indicates the time at which the capture began, in seconds
    # capture length, L1,           L2,         L3, ... # indicates the time it took to capture, in milliseconds
    #data,
    #1,              i1,    q1,    i2,   q2,   i3,   q3,  # iq data at t1 = 0
    #2,              i1,    q1,    i2,   q2,   i3,   q3,  # iq data at t2 = t1 + 1/sample_rate
    #3,              i1,    q1,    i2,   q2,   i3,   q3,  # iq data at t3 = t2 + 1/sample_rate
    # ... and so on and so forth

    OVERHEAD = 8
    n_per_shift = 1024
    file_path = "dump.csv"

    # start scanning!
    while not SDRWorker.cap(start,stop,numcaps,file_path,limplot):
        pass

    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)

        # get some metadata from the file
        metadata = returnMeta(file_path)
        captime = metadata[0]
        sample_rate = metadata[1]
        center_freq = metadata[2]
        freq_span = metadata[3]

        # find number of data points in the file
        numcols = max([int(s) for s in rows[4][1:]])
        numrows = len(rows) - OVERHEAD
        
        global data    
        data = np.empty(shape=(numrows,numcols),dtype=np.complex128)

        # fill the data array full of data
        for i in range(1,2*(numcols-1),2):
            for k in range(OVERHEAD,numrows,1):
                real = float(rows[k][i])
                imag = float(rows[k][i+1])    
                data[k][int(i/2)] = complex(real,imag)

    # find the number of sweeps and normresh the data
    sweeps = numcols/2

    
    data = normresh(data,n_per_shift,int(sweeps))

    # placeholder
    global time
    time = np.arange(123)

    #time = 
    
    # assemble frequency domain representation of the captured signal
    
    # Assuming the signal being broadcast is constant across the entire frequency sweep, we can add every 256 samples together to create a picture of the entire frequency spectrum
    
    # Perform FFT
    global fft
    fft = np.fft.fft(data)
    
    global freqs
    freqs = np.linspace(center_freq-freq_span/2,center_freq+freq_span/2,len(fft))   

    return True

def returnData():
    return [data,time,fft,freqs]
