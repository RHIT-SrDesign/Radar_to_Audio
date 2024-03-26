import numpy as np
from numpy.fft import fft
from tensorflow.keras.models import load_model

L=1024 # Length of data sequence given to model
D=100  # Decimation factor between short and long sequences

def characterize(path, data):
    model_modtype = load_model(path+'modtype_2.h5')
    model_freqagility = load_model(path+'freq_agility_2.h5')
    #model_pwagility = load_model(path+'pw_agility_2.h5') # Bad Model
    model_priagility = load_model(path+'pri_agility_2.h5')
    #model_pw = load_model(path+'pw_2.h5')
    #model_pri = load_model(path+'pri_2.h5')

    x = prepare_data_sequences(data)

    modtype = model_modtype.predict(x)[1]
    freqagility = model_freqagility.predict(x)[1]
    priagility = model_priagility.predict(x)[1]

    return [modtype, freqagility, priagility]

def prepare_data_sequences(input):
    data = np.zeros((1, L, 8))

    # Generate all of the features that the model will have available to train on
    x_short = input[np.arange(0, L-1, 1)]
    x_long = input[np.arange(0, (L-1)*D, D)]
    xf_short = fft(x_short)
    xf_long = fft(x_long)
    x_ms = abs(x_short)
    x_ps = np.angle(x_short)
    x_ml = abs(x_long)
    x_pl = np.angle(x_long)
    xf_ms = abs(xf_short)
    xf_ps = np.angle(xf_short)
    xf_ml = abs(xf_long)
    xf_pl = np.angle(xf_long)

    data[1,:,0] = x_ms # Time domain magnitude
    data[1,:,1] = x_ps # Time domain phase
    data[1,:,2] = x_ml # Time domain magnitude, decimated by 100
    data[1,:,3] = x_pl # Time domain phase, decimated by 100
    data[1,:,4] = xf_ms # FFT mag of 1&2
    data[1,:,5] = xf_ps # FFT phase of 1&2
    data[1,:,6] = xf_ml # FFT mag of 3&4
    data[1,:,7] = xf_pl # FFT phase of 3&4
    
    return data