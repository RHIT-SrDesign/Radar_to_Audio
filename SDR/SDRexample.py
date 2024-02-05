import SDRDriver
import matplotlib.pyplot as plt
import numpy as np


start = 2.0e9 # start at 2.4 GHz
end = 3.0e9 # end at 2.9 Ghz
numcaps = 100 # grab this signal 5 times
limplot = False # dont plot while capturing


def main():
    # run the SDR, wait for it to be done
    print("running sweeps")
    while not SDRDriver.runProc(start,end,numcaps,limplot):
        pass
    
    # retrieve the data from this sweep
    [data, time, fft, freqs] = SDRDriver.returnData()

    print("done running sweeps")


    # Plot the time domain data points
    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(data))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # find the PSD
    power_spectral_density = np.abs(fft)**2 / len(data)

    # Plot power spectral density in dB
    plt.plot(freqs, 10 * np.log10(power_spectral_density))
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.show()


if __name__ == main():
    main()