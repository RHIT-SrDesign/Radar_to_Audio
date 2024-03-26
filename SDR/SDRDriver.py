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

def remove_below_threshold(arr, threshold):
    magnitudes = arr  # Calculate magnitudes of complex numbers
    mask = magnitudes >= threshold  # Create a boolean mask for elements above threshold
    filtered_arr = arr[mask]  # Apply mask to filter elements
    return filtered_arr,mask

def sliding_window_average(data, window_size):
    # Pad the data array to handle edge cases
    padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
    
    # Create a view of the data with the rolling window
    shape = padded_data.shape[:-1] + (padded_data.shape[-1] - window_size + 1, window_size)
    strides = padded_data.strides + (padded_data.strides[-1],)
    windowed_data = np.lib.stride_tricks.as_strided(padded_data, shape=shape, strides=strides)
    
    # Calculate the average within each window
    window_averages = np.mean(windowed_data, axis=0)
    
    return window_averages

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
    file_path = "dump.csv"
    offset = 100e6
    threshold = -300


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

    # remove all elements below the threshold
    fft,mask = remove_below_threshold(fft, threshold)

    global data
    data = np.fft.ifft(fft,n_per_shift,norm="backward")

    print(np.shape(fft))

     # calculate how long in seconds we sample for in the SDR
    time_per_samp = n_per_shift/sample_rate
    # we now have this new higher effective sample rate which we can use to get the time series data
    new_sample_rate = np.shape(fft)[1]/time_per_samp

    print(new_sample_rate)

    global freqs
    freqs = np.linspace(center_freq-freq_span/2-offset,center_freq+freq_span/2-offset,np.shape(fft)[1]) 
    #freqs = np.fft.fftfreq(np.shape(fft)[1],1/new_sample_rate) + center_freq    
    
    freqs = freqs[mask]


    # create the time array
    global time
    time = np.array([])
    
    # we have 
    capnum = max(capnums)+1


    for i in range(capnum):
        starttime = captimes[i]

        endtime = starttime + n_per_shift/sample_rate

        littletime = np.arange(starttime,endtime,1/sample_rate)
        
        time = np.hstack((time,littletime)) 
        

    data = data.ravel()

    # remove all elements below the threshold
    data = remove_below_threshold(data, threshold)

    
    time = time[0:np.shape(data)[0]]

    # assemble frequency domain representation of the captured signal
    
    # Assuming the signal being broadcast is constant across the entire frequency sweep, we can add every 256 samples together to create a picture of the entire frequency spectrum

    return True

def getProc(file_path,n_per_shift):

    OVERHEAD = 8
    offset = 100e6
    threshold = -300
    window = 50


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

        # for each capture
        colnum = 0
        for i in range(1,2*(numcols)+1,2):

            littlearr = np.empty(shape=(numrows,1),dtype=np.complex128)
            
            # fill in the entire fft
            for k in range(OVERHEAD,numrows,1):
                real = float(rows[k][i])
                imag = float(rows[k][i+1])    
                littlearr[k] = complex(real,imag)

            # put the captured fft into fft array
            fft[colnum][:] = littlearr.T
            colnum = colnum + 1

    original = np.shape(fft)[1]   

    # perform n point averaging on the frequency data
    #fft = sliding_window_average(fft,window)

    # remove all elements below the threshold
    fft,mask = remove_below_threshold(fft, threshold)

    # Perform iFFT
    global data
    data = np.fft.ifft(fft,n_per_shift,norm="backward")

    # calculate how long in seconds we sample for in the SDR
    time_per_samp = n_per_shift/sample_rate
    # we now have this new higher effective sample rate which we can use to get the time series data
    new_sample_rate = original/time_per_samp

    print(new_sample_rate)

    global freqs
    freqs = np.linspace(center_freq-freq_span/2,center_freq+freq_span/2,original) 
    #freqs = sliding_window_average(freqs,window)

    freqs = freqs[mask]
 
    

    # create the time array
    global time
    time = np.array([])
    
    # we have 
    capnum = max(capnums)+1


    for i in range(capnum):
        starttime = captimes[i]

        endtime = starttime + n_per_shift/new_sample_rate

        littletime = np.arange(starttime,endtime,1/new_sample_rate)
        
        time = np.hstack((time,littletime)) 
        

    data = data.ravel()

    
    time = time[0:np.shape(data)[0]]

    # assemble frequency domain representation of the captured signal
    
    # Assuming the signal being broadcast is constant across the entire frequency sweep, we can add every 256 samples together to create a picture of the entire frequency spectrum

    return True

def getSpecificProc(file_path,n_per_shift,desiredcapnum,startsamp,endsamp):

    OVERHEAD = 8
    offset = 100e6
    threshold = -300

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

        # for each capture
        colnum = 0
        for i in range(1,2*(numcols)+1,2):

            littlearr = np.empty(shape=(numrows,1),dtype=np.complex128)
            
            # fill in the entire fft
            #TODO: at the beginning of every capture, there is a large DC spike, need to get rid of that
            for k in range(OVERHEAD,numrows,1):
                real = float(rows[k][i])
                imag = float(rows[k][i+1])    
                littlearr[k] = complex(real,imag)

            # put the captured fft into fft array
            fft[colnum][:] = littlearr.T
            colnum = colnum + 1

    # only grab top half of frequencies for analysis (2 radars) 
    print(np.shape(fft))     
    original = np.shape(fft)[1]  
    capnumprint = desiredcapnum
    #for i in range(original):
    #    print(fft[capnumprint][i])
    
    print(startsamp)
    print(endsamp)

    # only plot one cap
    fft = fft[capnumprint][startsamp:endsamp]
    print(np.shape(fft))   

    new = endsamp-startsamp

    # here remove all elements that are at the beginning of each capture

    # remove all elements below the threshold
    fft,mask = remove_below_threshold(fft, threshold)

    # Perform iFFT
    global data
    data = np.fft.ifft(fft,n_per_shift)

    # remove the first 15 and the last 15 elements from the array
    data = data[15:-15]


    # calculate how long in seconds we sample for in the SDR
    time_per_samp = n_per_shift/sample_rate
    # we now have this new higher effective sample rate which we can use to get the time series data
    new_sample_rate = new/time_per_samp

    print(new_sample_rate)

    global freqs
    freqs = np.linspace(center_freq-freq_span/2,center_freq+freq_span/2,original) 

    # cut the freqs in half too
    freqs = freqs[startsamp:endsamp]
    freqs = freqs[mask]

    # create the time array
    global time
    time = np.array([])
 
    starttime = 0

    endtime = starttime + n_per_shift/new_sample_rate

    time = np.arange(starttime,endtime,1/new_sample_rate)        

    
    data = data.ravel()
    
    time = time[15:-15]

    time = time[0:np.shape(data)[0]]


    # assemble frequency domain representation of the captured signal
    
    # Assuming the signal being broadcast is constant across the entire frequency sweep, we can add every 256 samples together to create a picture of the entire frequency spectrum

    return True

def returnData():
    return [data,time,fft,freqs]
