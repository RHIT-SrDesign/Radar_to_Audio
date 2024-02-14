import SDRDriver
import matplotlib.pyplot as plt
import numpy as np


filename = 'dump.csv'
n_per_shift = 102400

def main():
    # run the SDR, wait for it to be done
    print("running sweeps")
    while not SDRDriver.getProc(filename,n_per_shift):
        pass
    
    # retrieve the data from this sweep
    [data, time, fft, freqs] = SDRDriver.returnData()

    print("done running sweeps")


    # Plot the time domain data points
    plt.figure(figsize=(10, 6))
    plt.plot(time,np.abs(data))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot power spectral density in dB
    plt.plot(freqs, fft.T)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.show()


if __name__ == main():
    main()