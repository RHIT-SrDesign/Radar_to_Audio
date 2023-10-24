import sys,os
from pdb import set_trace

import numpy as np
import scipy.signal as sig
from matplotlib import pyplot as plt

from bluefile.bluefile import readheader,unpack_data_from_stream,_rep_tran

class MidasFile(object):
    def __init__(self,file_path):
        self.file_path = r"C:\Users\fennelj1\Desktop\radarAudio\SynthRad21\data\iq\angry_alpaca_sas.tmp"
        self._read_header()
        # 'size' is the number of elements in the file
        # 'bpe' is the number of bytes per element
        self.n_bytes_in_file = self.header['size'] * self.header['bpe']
        self.endian = _rep_tran[self.header['data_rep']]
        self.sample_rate = round(1/self.header['xdelta'])
        self.data_duration = self.header['size']*self.header['xdelta']
        try:
            self.fp = open(self.file_path,'rb')
        except Exception as e:
            print('Exception caught in MidasFile.__init__: {}'.format(e))
            sys.exit(1)

    def _read_header(self):
        self.header = readheader(self.file_path)

    def seek_to_time(self,t,whence=0):
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

        n_bytes_to_seek = int(self.seconds_to_elements(t) * self.header['bpe'])
        if whence == 0:
            # advance n_bytes_to_seek by header_size
            n_bytes_to_seek += self.header['data_start']
            n_bytes_to_seek = int(n_bytes_to_seek)
        try:
            self.fp.seek(n_bytes_to_seek,whence)
        except Exception as e:
            print('Exception caught in MidasFile.seek_to_time: {}'.format(e))

    def tell_time(self):
        try:
            t = self.fp.tell()/self.header['bpe']/self.sample_rate
        except Exception as e:
            print('Exception caught in MidasFile.tell_time: {}'.format(e))
            sys.exit(1)
        return t

    def seconds_to_elements(self,t):
        return int(t * self.sample_rate)

    def read_at_time(self,t,delta_t,reset_time=False):
        if reset_time:
            t0 = self.tell_time()
        n_elements = self.seconds_to_elements(delta_t)
        self.seek_to_time(t)
        data = next(self.read(n_elements))
        if reset_time:
            self.seek_to_time(t0)
        return data
        

    def read(self,n_elements,n_overlap=0):
        '''
        Generator to read from the file.
        n_elements and n_overlap are specified in terms of elements in the MIDAS file.
        '''
        n_elements = int(n_elements)
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
        # data = data[:,0] + 1j*data[:,1]
        if n_overlap > 0:
            self.fp.seek(-n_bytes_to_seek,1) # move pointer back according to the overlap amoutn
        return np.array(data)



if __name__=='__main__':
    # example usage
    file_path = 'file.tmp'
    print('Running tests on '+file_path+'...')
    mf = MidasFile(file_path)
    n_elements = 1e6 # read this many elements 
    n_overlap = int(n_elements/3) # overlap by a non-integer multiple just to check that things appear to be working
    n_fft = 1024*2
    window = sig.chebwin(n_fft,100,sym=False) # any window will probably do
    
    # # try reading an arbitrary window
    # d = mf.read_at_time(0.5,1e-3)
    # f,t,s = sig.stft(np.real(d),mf.sample_rate,window,nperseg=len(window),boundary=None)
    # s_disp = 20*np.log10(np.abs(s)/np.max(np.abs(s))) # make the spectrum easier to read
    # plt.figure()
    # plt.pcolormesh(t,f,s_disp)
    # plt.clim([-60, 0]) # set dynamic range
    # plt.colorbar()
    # plt.show()


    # reset the file
    mf.seek_to_time(0)
    print(mf._read_stream(n_elements,n_overlap=n_overlap))
    # loop throgh and read the whole thing
    # for d in mf.read(n_elements,n_overlap=n_overlap):
    #     f,t,s = sig.stft(np.real(d),mf.sample_rate,window,nperseg=len(window),boundary=None)
    #     s_disp = 20*np.log10(np.abs(s)/np.max(np.abs(s))) # make the spectrum easier to read
    #     print(f,t,s)
    #     print(s_disp)
    #     # plt.figure()
    #     # plt.clim([-60, 0]) # set dynamic range
    #     # plt.colorbar()
    #     # plt.show() # close the plot to continue

# test test test