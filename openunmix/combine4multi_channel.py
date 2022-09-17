#coding:utf-8

# combine four wav files into one multiple-channels wav file


import sys
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.18.4
#  scipy 1.4.1

def read_wav( file_path, MONO=False ):
    try:
        sr, w = wavread( file_path)
    except:
        print ('error: wavread ', file_path)
        sys.exit()
    else:
        w= w / (2 ** 15)
        if MONO and w.ndim == 2:  # if MONO is true and input is stereo, convert it to mono.
            w= np.average(w, axis=1)
            w= w.reshape([len(w),1])
        print ('sampling rate ', sr)
        print ('size', w.shape)
    return w, sr


# To write multiple-channels, use a 2-D array of shape (Nsamples, Nchannels).

def save_wav( file_path, data, sr=48000):
    amplitude = np.iinfo(np.int16).max
    try:
        wavwrite( file_path , sr, np.array( amplitude * data , dtype=np.int16))
    except:
        print ('error: wavwrite ', file_path)
        sys.exit()
    print ('wrote ', file_path)


if __name__ == '__main__':
    
    dir1='sample/'
    flist1=['bass.wav','drumes.wav','vocals.wav','others.wav']
    outFile='sample_combined.wav'
    wlist=[]
    
    for fi in flist1:
        w,sr1= read_wav( dir1 + fi) #, MONO=True)  # specify MONO=True, if input wav becomes MONO.
        wlist.append(w)
        
    
    y= np.hstack( (wlist[0], wlist[1], wlist[2], wlist[3]) )
    print ('y.shape', y.shape)
    save_wav(outFile, y, sr1)
