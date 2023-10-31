import sys,os
from pdb import set_trace

import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt

from bluefile.bluefile import readheader,unpack_data_from_stream,_rep_tran

class MidasFile(object):
    def __init__(self,file_path):
        self.file_path = file_path
        self._read_header()
        # 'size' is the number of elements in the file
        # 'bpe' is the number of bytes per element
        self.n_bytes_in_file = self.header['size'] * self.header['bpe'] #This value is read in from the header block as per MIDAS blue file standard
        self.endian = _rep_tran[self.header['data_rep']] #This value is read in from the header block as per MIDAS blue file standards
        self.sample_rate = round(1/self.header['xdelta']) #This value reads in the interval between consecutive samples and finds the reciporical likely measured in Hz
        self.data_duration = self.header['size']*self.header['xdelta'] #Reads in the number of data elements and multiplies by the interval between consecutive samples
        self.n_elements = self.header['size']
        try:
            self.fp = open(self.file_path,'rb') #Opens the file in read mode with binary I/O
        except Exception as e:
            print('Exception caught in MidasFile.__init__: {}'.format(e)) #if the file doesn't open prints an error and gracefully exits
            sys.exit(1)

    def _read_header(self): #function read header uses the read header helper function at the location of the file
        self.header = readheader(self.file_path)

    def seek_to_time(self,t,whence=0): #jumps to element at this time event and starts from there
        ''' 
        WARNING:
        The try-except block will execute silently if t > tmax (i.e., we ask to
        seek past the end of the file.) It just quietly puts the poitner at the
        end of the file. I guess this is expected behavior from seek, so we'll
        just go along with it.

        Usage:
        seek to the point (sample) in the file nearest to t (in seconds)
        'whence' follows seek... i.e., 
        whence=0: beginning of file
        whence=1: current position
        whence=2: end of file
        '''

        n_bytes_to_seek = int(self.seconds_to_elements(t) * self.header['bpe']) #takes the time input, runs seconds to elements function, and multiplies by the bytes per element
        if whence == 0: #if seeking from the beggining of file
           
            n_bytes_to_seek += self.header['data_start'] #start all seek requests at the base location of where the data starts
            n_bytes_to_seek = int(n_bytes_to_seek) #cast the number of bytes as an integer
        try:
            self.fp.seek(n_bytes_to_seek,whence) #run the seek command to the byte in the file
        except Exception as e:
            print('Exception caught in MidasFile.seek_to_time: {}'.format(e)) #catch error, fail gracefully

    def tell_time(self): #this function returns the time of the cursor in the file 
        try:
            t = self.fp.tell()/self.header['bpe']/self.sample_rate #sets time equal to the current curosr location(integer index?), divided by the bytes per element, divided by the samples per second
        except Exception as e:
            print('Exception caught in MidasFile.tell_time: {}'.format(e)) #If this doesn't work handles error greacefully
            sys.exit(1)
        return t

    def seconds_to_elements(self,t): #converts current time to element of the data
        return int(t * self.sample_rate) #multiplies time by sample rate to get element number and casts as int

    def read_at_time(self,t,delta_t,reset_time=False): 
        if reset_time: #if function takes in reset time as true
            t0 = self.tell_time() #Reads the current time
        n_elements = self.seconds_to_elements(delta_t) #sets n elements equal to seconds to elements of the time between samples
        self.seek_to_time(t) #jump to the requested time in the function input
        data = next(self.read(n_elements)) #sets the variable data to the result of the next function, of the total number of elements
        if reset_time:
            self.seek_to_time(t0) #sets time all the way back to 0
        return data
        

    def read(self,n_elements,n_overlap=0): #Takes in the number of elemebts
        '''
        Generator to read from the file.
        n_elements and n_overlap are specified in terms of elements in the MIDAS file.
        '''
        n_elements = int(n_elements) #casts number of elements to read as an integer
        while self.fp.tell() < self.n_bytes_in_file:
            yield self._read_stream(n_elements,n_overlap)
                
    def _read_stream(self,n_elements,n_overlap):
        if n_overlap > 0:
            n_bytes_to_seek = int(n_overlap * self.header['bpe'])
        try:
            data = unpack_data_from_stream(self.header,self.fp,elements=n_elements,endian=self.endian)
        except Exception as e:
            raise(e)
        data = np.array(data)
        mag = data
        # mag = np.abs(data)
        if n_overlap > 0:
            self.fp.seek(-n_bytes_to_seek,1) # move pointer back according to the overlap amoutn
        # return np.array(data)
        return mag
