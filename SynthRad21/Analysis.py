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

def execute(file ): #= r"SynthRad21\angry_bear_sas.tmp",
    file_path = file
    print('Running tests on '+file_path+'...')
    mf = midas.MidasFile(file_path)
    n_elements = mf.n_elements # read this many elements 
    n_overlap = int(n_elements/2) # overlap by a non-integer multiple just to check that things appear to be working
    n_fft = 1024*2
    window = sig.chebwin(n_fft,100,sym=False) # any window will probably do cheb window with 100db range, not rly sure why tho :/

    # reset the file
    mf.seek_to_time(0)
    npData = mf._read_stream(n_elements,n_overlap=n_overlap)
    # npData = mf.read(n_elements,n_overlap=0)
    print("data read")
    print("sample rate is {} Hz".format(mf.sample_rate))
    index = np.array(range(mf.n_elements))
    elements = index
    index = index/mf.sample_rate    
    #nfft = int(pow(2, np.ceil(np.log(len(elements))/np.log(2))))
    #establish graphic frame


    print("computing fft")
    #Calculate fft information on signal
    N = (len(elements))
    #nfft = nfft // 1024
    nfft = 2048
    fftData=np.fft.fft(npData)
    fftFreq = np.fft.fftfreq(N,1/mf.sample_rate)
    fftData = fftData*(1/mf.sample_rate)
    cf = "?"
    txt = "Frequency centered around {center}"
    txt=txt.format(center=cf)
    frq_amplitude = (1/N) * np.abs(fftData)
    frq_amplitude_db = 20*np.log10(frq_amplitude)
    avgPow= np.average(frq_amplitude_db)
    maxPow= np.max(frq_amplitude_db)
    print("plotting")

    #


    fig, (ax) = plt.subplots(nrows=1)
    ax.plot(fftFreq,frq_amplitude_db)
    
    # ax.specgram(npData,nfft,mf.sample_rate, sides='onesided')
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel("Frequency [Hz]")    
    ax.set_xlabel(txt)
    ax.set_ylabel("Amplitude")    
    plt.close('all')
    print("plot ready")
    # gc.collect()
    # fig, (ax) = plt.subplots(nrows=1)
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Tens of MHz")
    # bounds = [avgPow,maxPow]
    # npData = npData[::100]
    # f, t, Sxx = sig.spectrogram(npData,mf.sample_rate)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # Pxx, freqs, bins, im = ax.specgram(npData,nfft,mf.sample_rate, sides='onesided')
    # fig.colorbar().set_label('Intensity [dB]')
    # fig.clim(-280,-140)
    
    fs = mf.sample_rate
    wav.signal2wav(fs,npData)
    
    return fig


    
    