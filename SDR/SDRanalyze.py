import SDRDriver
import matplotlib.pyplot as plt
import numpy as np


filename = 'AirportDoppler4.csv'
n_per_shift = 102400

def main():
    # run the SDR, wait for it to be done
    print("running sweeps")
    while not SDRDriver.getProc(filename,n_per_shift):
        pass
    segstart = 0
    segend = 5

    capnumlookat = 0
    startsamp = n_per_shift*(segstart)
    endsamp = n_per_shift*segend
    #while not SDRDriver.getSpecificProc(filename,n_per_shift,capnumlookat,startsamp,endsamp):
    #    pass
    
    # retrieve the data from this sweep
    [data, time, fft, freqs] = SDRDriver.returnData()

    print("done running sweeps")

    fig,ax = plt.subplots(2)
    # Plot the time domain data points

    ax[0].plot(time*1000,np.abs(data),'.')
    #ax[0].plot(np.abs(data),'.')
    ax[0].set_title('Time Domain')
    ax[0].set_xlabel('Time (ms)')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].grid(True)

    


    ax[1].plot(freqs, fft.T)
    ax[1].set_title('Power Spectral Density')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power/Frequency (dB/Hz)')
    ax[1].grid(True)
    
    fig.tight_layout()
    plt.show()

if __name__ == main():
    main()