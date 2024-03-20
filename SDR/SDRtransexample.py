import SDRDriverTrans
import matplotlib.pyplot as plt
import numpy as np


start = 0.07e9 # start at 2.4 GHz
end = 0.5e9 # end at 2.9 Ghz
numcaps = 1 # grab this signal 5 times
limplot = True # dont plot while capturing


def main():
    # run the SDR, wait for it to be done
    print("running sweeps")
    while not SDRDriverTrans.runProc(start,end,numcaps,limplot):
        pass
    
    # retrieve the data from this sweep
    [data, time, fft, freqs] = SDRDriverTrans.returnData()

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
    #plt.ylim(top=100,bottom=30)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.show()


if __name__ == main():
    main()