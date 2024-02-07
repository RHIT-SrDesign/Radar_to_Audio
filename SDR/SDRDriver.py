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

        # grab the capnums, captimes, and the cap durations
        capnums = np.array(rows[4][1:],dtype=int)
        captimes = np.array(rows[5][1:],dtype=float)
        capdurs = np.array(rows[6][1:],dtype=float)


        # get the start times and the stop times and return them as row vectors
        # here is where id do it [int(s) for s in rows[4][1:]]

    return [captime,sample_rate,center_freq,freq_span,capnums,captimes,capdurs] 

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
    n_per_shift = 102400
    file_path = "AirportDoppler4.csv"

    # start scanning!
    while not SDRWorker.cap(start,stop,n_per_shift,numcaps,file_path,limplot):
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
        
        # get the time arrays from the file
        capnums = metadata[4]
        captimes = metadata[5]
        capdurs = metadata[6]

        # find number of data points in the file
        numcols = max([int(s) for s in rows[4][1:]]) + 1
        numrows = len(rows) - OVERHEAD
        
        # store the fft result in the columns, and each additional cap fft in the rows
        global fft    
        fft = np.empty(shape=(numcols,numrows),dtype=np.complex128)

        # fill the data array full of data
        colnum = 0
        for i in range(1,2*(numcols)+1,2):

            littlearr = np.empty(shape=(numrows,1),dtype=np.complex128)

            for k in range(OVERHEAD,numrows,1):
                real = float(rows[k][i])
                imag = float(rows[k][i+1])    
                littlearr[k] = complex(real,imag)

            fft[colnum][:] = littlearr.T
            colnum = colnum + 1

    global freqs
    freqs = np.linspace(center_freq-freq_span/2,center_freq+freq_span/2,np.shape(fft)[1]) 

    # Perform iFFT
    global data
    data = np.fft.ifft(fft)
    

    # calculate how long in seconds we sample for in the SDR
    time_per_samp = n_per_shift/sample_rate
    # we now have this new higher effective sample rate which we can use to get the time series data
    new_sample_rate = np.shape(data)[1]/time_per_samp

    print(new_sample_rate)
    # create the time array
    global time
    time = np.array([])
    
    # we have 
    capnum = max(capnums)+1
    for i in range(capnum):
        starttime = captimes[i]

        endtime = starttime + n_per_shift/sample_rate

        littletime = np.arange(starttime,endtime,1/new_sample_rate)
        
        time = np.hstack((time,littletime)) 

    
    
    data = data.ravel()
    time = time[0:np.shape(data)[0]]

    # assemble frequency domain representation of the captured signal
    
    # Assuming the signal being broadcast is constant across the entire frequency sweep, we can add every 256 samples together to create a picture of the entire frequency spectrum

    return True

def getProc(file_path,n_per_shift):

    OVERHEAD = 8


    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)

        # get some metadata from the file
        metadata = returnMeta(file_path)
        captime = metadata[0]
        sample_rate = metadata[1]
        center_freq = metadata[2]
        freq_span = metadata[3]
        
        # get the time arrays from the file
        capnums = metadata[4]
        captimes = metadata[5]
        capdurs = metadata[6]

        # find number of data points in the file
        numcols = max([int(s) for s in rows[4][1:]]) + 1
        numrows = len(rows) - OVERHEAD
        
        # store the fft result in the columns, and each additional cap fft in the rows
        global fft    
        fft = np.empty(shape=(numcols,numrows),dtype=np.complex128)

        # fill the data array full of data
        colnum = 0
        for i in range(1,2*(numcols)+1,2):

            littlearr = np.empty(shape=(numrows,1),dtype=np.complex128)

            for k in range(OVERHEAD,numrows,1):
                real = float(rows[k][i])
                imag = float(rows[k][i+1])    
                littlearr[k] = complex(real,imag)

            fft[colnum][:] = littlearr.T
            colnum = colnum + 1

    global freqs
    freqs = np.linspace(center_freq-freq_span/2,center_freq+freq_span/2,np.shape(fft)[1]) 

    # Perform iFFT
    global data
    data = np.fft.ifft(fft)
    

    # calculate how long in seconds we sample for in the SDR
    time_per_samp = n_per_shift/sample_rate
    # we now have this new higher effective sample rate which we can use to get the time series data
    new_sample_rate = np.shape(data)[1]/time_per_samp

    print(new_sample_rate)
    # create the time array
    global time
    time = np.array([])
    
    # we have 
    capnum = max(capnums)+1
    for i in range(capnum):
        starttime = captimes[i]

        endtime = starttime + n_per_shift/sample_rate

        littletime = np.arange(starttime,endtime,1/new_sample_rate)
        
        time = np.hstack((time,littletime)) 

    
    
    data = data.ravel()
    time = time[0:np.shape(data)[0]]

    # assemble frequency domain representation of the captured signal
    
    # Assuming the signal being broadcast is constant across the entire frequency sweep, we can add every 256 samples together to create a picture of the entire frequency spectrum

    return True

def returnData():
    return [data,time,fft,freqs]
