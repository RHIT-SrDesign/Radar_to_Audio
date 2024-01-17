import numpy as np
import pandas as pd
from midas_tools import MidasFile
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy import signal



sample_rate = 50e6 # samples per second
plutomax = 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs


# Create transmit waveform (QPSK, 16 samples per symbol)
num_symbols = 1000
x_int = np.random.randint(0, 4, num_symbols) # 0 to 3
x_degrees = x_int*360/4.0 + 45 # 45, 135, 225, 315 degrees
x_radians = x_degrees*np.pi/180.0 # sin() and cos() takes in radians
x_symbols = np.cos(x_radians) + 1j*np.sin(x_radians) # this produces our QPSK complex symbols
num_sampspersymb = sample_rate/num_symbols
samples = np.repeat(x_symbols, num_sampspersymb) # num_sampspersymb samples per symbol (rectangular pulses)
samples *= plutomax # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs


# test the QPSK symbols that guaranteed work
filename = "QPSK.csv"
df=pd.DataFrame(samples.ravel())
df.to_csv(filename)

# now time for the real file
importfilename = "angry_bear_sas.iq"
mf = MidasFile(importfilename)

# arbitrary start and stop time
data = mf.read_at_time(0,1)

# correct for differences in sample rate
orig_len = data.shape[0]

samp_correction = sample_rate/orig_len
num_samples_resampled = int(orig_len * samp_correction)

data = resample(data, num_samples_resampled)
print("done with resampling")

datamax = np.max(data) # find the max of the data
#datamax = 0.0003

data *= plutomax/datamax # scale data to be between 2**14 and -2**14
print("done with rescaling")


# Calculate the frequency plots
psd = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2
psd1_dB = 10*np.log10(psd)
f1 = np.linspace(0,2*sample_rate,len(psd1_dB))

psd = np.abs(np.fft.fftshift(np.fft.fft(data)))**2
psd2_dB = 10*np.log10(psd)
f2 = np.linspace(0,2*sample_rate,len(psd2_dB))


# save the experimental data
#filename = "real.csv"
#df=pd.DataFrame(data.ravel())
#df.to_csv(filename)

#print("done saving")

# plot the data
plt.figure()

plt.plot(data)
plt.plot(samples)

plt.figure()
plt.plot(f1,psd1_dB)

plt.figure()
plt.plot(f2,psd2_dB)


plt.show() # close the plot to continue
