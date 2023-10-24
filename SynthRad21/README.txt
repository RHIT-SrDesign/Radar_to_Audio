# SynthRad-21
SynthRad-21 is a dataset containing 25 simulated single-function radars. Each emitter is provided as an individual, noiseless, complex-valued baseband recording in the Midas1000 format. It is up to the end user to implement any necessary preprocessing, including upsampling, shifting to the correct IF center frequency, and adding noise and other impairments. 

## Reading Midas1000 Files
The file 'midas_tools.py' and its dependency, the directory 'bluefile', are provided to read Midas1000 files into Python.

```python
from midas_tools import MidasFile

mf = MidasFile(path_to_file)

sample_rate_hz = mf.sample_rate
data_duration_sec = mf.data_duration
file_size = mf.n_bytes_in_file

data = mf.read_at_time(start_time, window_length_sec, reset_time=True)

mf.fp.close()
```

Alternatively, the file 'read_midas_1xxx.m' is provided to read Midas1000 files into MATLAB.

## Data Preprocessing
The data_preprocessing.py script is something I threw together to do all the necessary preprocessing to create a set of training and testing examples for my research. It is a mess because I was scrambling to get it out to a few students I have been working with... So I apologize for that.

In my work I am applying CNNs to spectrograms (magnitude squared of the short time Fourier transform) of passively intercepted radar signals, with the goal of classifying the emitter (which animal). Additionally, we have been looking into open set recognition (identifying unknown classes without interfering with classification of known classes), and multi-target detection and classification using computer vision techniques. As a result, the data preprocessing script was primarily intended to produce spectrograms and corresponding annotation files, each of which contains bounding boxes to localize the signal within a spectrogram, and metadata with information about the spectrogram (including time and frequency resolution, the sample rate, the FFT size used, etc.), saved as a json file.

As an alternative to the real-valued spectrogram, there is a flag to output the complex-valued STFT, or the raw sampled signal as output. Two important warnings! First, I haven't tested these settings, so there may be minor errors in how the data is saved. Second, the STFT outputs are currently being saved as mat files, and the raw data is currently being saved as npy files. Both result in **much** larger file sizes than the spectrograms, which are saved as PNG files!! Might need to reduce the number of examples per class accordingly.

One additional note, the data was originally intended to span a 1 GHz receiver bandwidth, with emitters occupying the 8-9 GHz range. However, you could reduce the receiver bandwidth and scale the emitter center frequencies, accordingly. This could be of interest if, for example, you find the train/test examples at a 1 GHz sample rate are taking up too much hard disk space.

### Preprocessor Parameters
These can be adjusted via the command line interface, or within the script, where they are provided as defaults to the CLI parser.
-	in_root_path: String, path to location containing the 'pdw', 'iq', and 'config' directories.
-	out_root_path: String, path to location where train, validate, and test directories will be placed.
-	data_type: String, one of [“spectrogram”, “stft”, “raw”]. Determines how data examples will be output.
-	soi_list: List of strings, signals of interest (filenames without extensions), for which examples will be generated. If empty, will use all files in 'iq' and 'pdw' directories.
-	exclude_list: List of strings, emitters to ignore when generating examples. Useful if there are only one or two emitters to be ignored.
-	window_len_sec: Float, amount of time spanned by each example, in seconds.
-	seg_len_samp: Integer, length of each STFT subsegment. Ignored if output is “raw”.
-	fft_size: Integer, length of FFT applied to each STFT subsegment. Ignored if output is “raw”.
-	snr_list: List of floats, SNR levels at which to generate examples.
-	output_sample_rate: Float, the sample rate corresponding with the generated data. For the angry animals, this is 1e9 Hz.
-	num_per_class_per_snr: Integer, the number of examples to generate for each SNR level for each class.
-	pct_train: Float, a number between 0 and 1 indicating the percentage of examples to place in the training set.
-	pct_validate: Float, a number between 0 and 1 indicating the percentage of examples to place in the validation set.
-	pct_test: Float, a number between 0 and 1 indicating the percentage of examples to place in the test set.
-	pct_two_signal: Float, a number between 0 and 1 indicating the number of examples containing two signals. Only applicable for the detection task.
-	pct_three_signal: Float, a number between 0 and 1 indicating the number of examples containing three signals. Only applicable for the detection task.
-	task: String, one of [“classification”, “classification_detection”,  “detection”]. The classification task will generate examples of individual emitters with noise added. The detection task will generate examples with multiple emitters present.

## Understanding the Radar Parameters
Each simulated radar recording was generated using the DEWM simulator. The DEWM configuration files for each emitter are distributed with the data, and are used in my preprocessing script to identify the correct center frequency for each upsampled waveform. However, these files can also be used to understand your results, for instance to understand why a classifier is unable to discriminate between two emitters.

Each emitter config file contains one or more lines with the following information:

Nreps | PRI (s) | Amp (Vp) | Freq (Hz) | PW (s) | Phase (rad) | Mode

For instance, the Angry Cheetah uses the following schedule:

DATA GLOBAL TransmitEventEntry angry_cheetah_sas Search MPRF
    20, 2.33e-3, 4.0e3, 8.246e9, 88e-6, 0, Pulse
    35, 2.33e-3, 4.0e3, 8.316e9, 88e-6, 0, Pulse

So, the cheetah will play 20 pulses at 8246 MHz, then 35 pulses at 8316 MHz. Both waveforms share the same amplitude, pulse width, and PRI, and both use unmodulated pulses. After the simulator reaches the end of the schedule, it will loop back to the beginning. So, the radar will repeat this pattern for the entire simulation. This dataset does not contain any multifunction or adaptive radars. Although the data contains radar waveforms with agilities, none of the radars adapt or change their behavior during the simulation.


## Contact Information
Please don't hesitate to reach out if you have any questions or find any issues with the code.

Chris Ebersole
AFRL/RYWE
christopher.ebersole.1@us.af.mil
