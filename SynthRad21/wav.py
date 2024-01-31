import scipy.io
from scipy import signal
import scipy.io.wavfile as wavf
import numpy as np

#def signal2wav (fs,signal,file = 'out.wav'):   
 #   wavf.write(file,fs,signal.astype(np.)) #need to fix write command (data type error)
    # wavf.read(signal)

def signal2wav(fs,npData):
    # Define parameters
    fs_original = fs  # Original sampling rate (Hz)
    fs_decimated = 50000      # Decimated sampling rate (Hz)
    center_freq = 2000        # Center frequency after upconversion (Hz)
    desired_bandwidth = 1000  # Desired bandwidth after compression (Hz)

    #compute the magnitude of the signal
    mag_signal = np.abs(npData) 

    # Decimate the signal
    decimation_factor = int(fs_original / fs_decimated)
    decimated_signal = mag_signal[::decimation_factor]

    # Upconvert the center frequency
    t = np.arange(len(decimated_signal)) / fs_decimated
    upconverted_signal = decimated_signal * np.cos(2 * np.pi * center_freq * t)

    # Compress the bandwidth using a low-pass filter
    nyquist_freq = fs_decimated / 2
    desired_cutoff = desired_bandwidth / 2
    filter_order = 100  # Adjust the filter order as needed
    b = signal.firwin(filter_order + 1, desired_cutoff / nyquist_freq)
    compressed_signal = signal.lfilter(b, 1.0, upconverted_signal)

    # Normalize the signal to avoid clipping
    compressed_signal /= np.max(np.abs(compressed_signal))

    # Convert to a WAV file
    wavf.write('audio.wav', fs_decimated, compressed_signal.astype(np.float32))

    # Display a message
    print(f'Audio has been saved to audio.wav.')