
import sys,os
from pdb import set_trace
import time
import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt
import midas_tools as midas

file_path = r"SynthRad21\angry_bear_sas.tmp"
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
nfft = int(pow(2, np.ceil(np.log(len(elements))/np.log(2))))
#establish graphic frame


print("computing fft")
#Calculate fft information on signal
N = (len(elements))
nfft = nfft // 1024
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


plt.subplot(2,1,1)
plt.plot(fftFreq,frq_amplitude_db)
plt.xlabel(txt)
plt.ylabel("Amplitude")
ax=plt.subplot(2,1,2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Tens of MHz")
bounds = [avgPow,maxPow]
Pxx, freqs, bins, im = plt.specgram(npData,nfft,mf.sample_rate ,noverlap = 9*nfft//10, mode = "magnitude",cmap="gnuplot")
plt.colorbar().set_label('Intensity [dB]')
plt.clim(-280,-140)
# plt.subplot(3,1,3, markevery=25)
# x = index[0::10]
# y = (np.abs(npData))[0::10]
# plt.scatter(x,y)
plt.show()  
