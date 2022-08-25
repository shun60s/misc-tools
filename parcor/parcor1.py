#coding:utf-8

#
#  k(PARCOR)係数と断面積？?を求める
#

import sys
import copy


import scipy.signal
from scipy.io.wavfile import read as wavread
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

from lpc_k  import *

# check version
#  python 3.6.4 on win32
#  numpy 1.18.4
#  scipy 1.4.1
#  matplotlib 3.3.1


class Class_FrameProcess(object):
    def __init__(self,wav_file, NFRAME_time=25, NSHIFT_time=10, lpcOrder=32, Delta_Freq=1,f_max=6000, specified_frame=None):
        # load wav file
        self.path0= wav_file
        self.yg, self.sr=self.load_wav( self.path0)
        print ('wav max value=', max(self.yg))
        
        self.NFRAME_time=NFRAME_time  # frame length unit [ms]      # 640 sr=16Khz 40mS  # 400 sr=16Khz 25mS 
        self.NSHIFT_time=NSHIFT_time  # shift length unit [ms]       # 320 sr=16Khz 20mS  # 160 sr=16khz 10mS
        self.lpcOrder=lpcOrder
        self.NFRAME= int( self.NFRAME_time / 1000 * self.sr)
        self.NSHIFT= int( self.NSHIFT_time / 1000 * self.sr)
        self.window = np.hamming(self.NFRAME)  # Windows is Hamming
        self.preemph=0.97  # pre-emphasis 
        
        self.Delta_Freq = Delta_Freq  # frequency resolution to compute frequency response
        self.f_max= f_max   # minimum frequency to detect formant
        self.FreqPoints_list= np.arange(0,self.f_max+self.Delta_Freq, self.Delta_Freq) * 2.0 * np.pi / self.sr
        
        self.count= int(((len(self.yg) - ( self.NFRAME - self.NSHIFT)) / self.NSHIFT))
        print ('count ', self.count)
        
        self.do_process(specified_frame)
    
    def do_process(self, specified_frame=None):
        #
        pos = 0  # position
        countr=0
        
        for loop in range(self.count):
            if specified_frame is not None:
                if loop != specified_frame:
                    continue
            ## copy to avoid original over-change
            frame = self.yg[pos:pos + self.NFRAME].copy()
            pos += self.NSHIFT
            
            ## pre-emphasis
            frame -= np.hstack((frame[0], frame[:-1])) * self.preemph
            ## do window
            windowed = self.window * frame
            ## get lpc coefficients
            a,e,k=lpc_k(windowed, self.lpcOrder)
            
            ## get area from k
            A=get_A(k)
            
            #
            print ('')
            print ('+ frame no.', loop)
            print ('lpc order ', self.lpcOrder)
            for l in range(len(k)):
                print('k[' + str(l) + ']', k[l])
            for l in range(len(A)):
                print('A[' + str(l) + ']', A[l])
            #
            self.do_process2(a,e,A)
    
    def do_process2(self, a,e,A):
        ## get lpc spectrum
        w, h = scipy.signal.freqz(np.sqrt(e), a, self.FreqPoints_list)  # from 0 to  f_max
        lpcspec = np.abs(h)
        
        lpcspec[lpcspec < 1.0e-6] = 1.0e-6  # to avoid log(0) error
        loglpcspec = 20 * np.log10(lpcspec)
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Amplitude [dB]')
        ax1.plot(self.FreqPoints_list, loglpcspec, 'b', ms=2)
        
        plt.grid()
        
        #---
        ax2 = fig.add_subplot(212)
        ax2.plot(A)
        
        #ax2.set_title('cross-section area')
        plt.xlabel('step-length   right(input) left(output)')
        plt.ylabel('Cross-section area [ratio]')
        plt.grid()
        plt.tight_layout()
        
        
        #plt.clf()
        #plt.close()
        plt.show()
        
        
    
    def load_wav(self, path0):
        # return 
        #        yg: wav data (mono) 
        #        sr: sampling rate
        try:
            sr, y = wavread(path0)
        except:
            print ('error: wavread ', path0)
            sys.exit()
        else:
            yg= y / (2 ** 15)
            if yg.ndim == 2:  # if stereo
                yg= np.average(yg, axis=1)
        
        print ('file ', path0)
        print ('sampling rate ', sr)
        print ('length ', len(yg))
        return yg,sr



if __name__ == '__main__':
    #
    CFP= Class_FrameProcess('wav/a_1-16k.wav', lpcOrder=32, specified_frame=14)
    #CFP= Class_FrameProcess('wav/a_1-16k.wav', lpcOrder=100, specified_frame=14)
    #CFP= Class_FrameProcess('wav/a_1-16k.wav', lpcOrder=13, specified_frame=14)
