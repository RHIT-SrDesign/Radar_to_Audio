import io
import sys,os
import gc
import wav
from pdb import set_trace
import time
import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from scipy.fft import fftshift
import midas_tools as midas

def replace_zero_magnitude(arr):
   
    magnitudes = np.abs(arr)  # Calculate magnitudes of complex numbers
    nonzero_magnitudes = magnitudes[magnitudes != 0]  # Filter out zero magnitudes
    if len(nonzero_magnitudes) == 0:
        return None  # If all magnitudes are zero, return None
    else:
        min_magnitude= np.min(nonzero_magnitudes)  # Index of minimum magnitude
        

    zero_mask = magnitudes == 0  # Mask for zero magnitudes
    arr[zero_mask] = min_magnitude  # Replace zero magnitudes with small number
    return arr

def execute(file ): #= r"SynthRad21\angry_bear_sas.tmp",
    file_path = file
    print('Running tests on '+file_path+'...')
    mf = midas.MidasFile(file_path)
    n_elements = mf.n_elements # read this many elements 
    n_overlap = int(n_elements/2) # overlap by a non-integer multiple just to check that things appear to be working
    #n_fft = 1024*2
    #window = sig.chebwin(n_fft,100,sym=False) # any window will probably do cheb window with 100db range, not rly sure why tho :/

    # reset the file
    mf.seek_to_time(0)
    npData = mf._read_stream(n_elements,n_overlap=n_overlap)

    npData = replace_zero_magnitude(npData)

    print("data read")
    print("sample rate is {} Hz".format(mf.sample_rate))
    #index = np.array(range(mf.n_elements))
    #elements = index
    #index = index/mf.sample_rate    
    #nfft = int(pow(2, np.ceil(np.log(len(elements))/np.log(2))))
    #establish graphic frame


    #print("computing fft")
    #Calculate fft information on signal
    #N = (len(elements))
    #nfft = nfft // 1024
    nfft = 2048*100
    #fftData=np.fft.fft(npData)
    #fftFreq = np.fft.fftfreq(N,1/mf.sample_rate)
    #fftData = fftData*(1/mf.sample_rate)
    #cf = "?"
    #txt = "Frequency centered around {center}"
    #txt=txt.format(center=cf)
    #frq_amplitude = (1/N) * np.abs(fftData)
    #frq_amplitude_db = 20*np.log10(frq_amplitude)
    #avgPow= np.average(frq_amplitude_db)
    #maxPow= np.max(frq_amplitude_db)
    print("plotting")

    #


    #fig, (ax) = plt.subplots(nrows=1)
    #ax.plot(fftFreq,frq_amplitude_db)
    Fs = mf.sample_rate  # the sampling frequency

    fig, (ax1) = plt.subplots(nrows=1)
    print(np.shape(npData))
    Pxx, freqs, bins, im = ax1.specgram(npData, NFFT=nfft, Fs=Fs)
    
    ax1.set(title='Specgram')
    fig.colorbar(im, ax=ax1)
    ax1.set_xlabel('Time [sec]')
    ax1.set_ylabel("Frequency [Hz]")    
    
    #ax2.plot(freqs,Pxx)

    #ax3.plot(np.abs(npData))


    plt.close('all')
    print("plot ready")
    
    wav.signal2wav(Fs,npData)
    
    return fig



    
    