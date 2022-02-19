#coding:utf-8

#  separate stereo wav  to mono L wav  and  mono R wav
#  and, convert to 16k sampling


import sys
import pathlib
import glob

import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite

import librosa



def change_suffix(file_name, file_out, new_suffix="_1"):
    # ファイルの拡張子を得る
    sf = pathlib.PurePath(file_name).suffix
    # ファイル名(拡張子除く)を得る
    st = pathlib.PurePath(file_name).stem
    
    return file_out + st + new_suffix + sf


def read_wav( file_path ):
    try:
        sr, w = wavread( file_path)
    except:
        print ('error: wavread ', file_path)
        sys.exit()
    else:
        w= w / (2 ** 15)
        print ('file_path', file_path)
        print ('sampling rate ', sr)
        print ('shape', w.shape)
    return w, sr

def save_wav( file_path, data, sr):
    amplitude = np.iinfo(np.int16).max
    try:
        wavwrite( file_path , sr, np.array( amplitude * data , dtype=np.int16))
    except:
        print ('error: wavwrite ', file_path)
        sys.exit()
    print ('wrote ', file_path)


def get_wav_files( path ):
    # wavファイルのリストを作成する
    return glob.glob( path + "*.wav")

if __name__ == '__main__':
    
    
    file_path= "WAV/"               # specify input stereo wav folder
    file_out=  "WAV_MONO/"          # specify output mono wav folder
    file_out_16k=  "WAV_MONO_16K/"  # specify output mono 16khz sampling wav folder
    sr16k=16000
    
    wav_files_list=get_wav_files(file_path)
    
    
    for file_path_in in wav_files_list:
        
        
        w,sr= read_wav( file_path_in)
        
        
        file_path_0= change_suffix(file_path_in, file_out, new_suffix="_L")
        save_wav(file_path_0, w[:,0], sr=sr)
        file_path_1= change_suffix(file_path_in, file_out, new_suffix="_R")
        save_wav(file_path_1, w[:,1], sr=sr)
        
        # convert to 16k sampling
        file_path_0= change_suffix(file_path_in, file_out_16k, new_suffix="_16k_L")
        y0=librosa.resample(w[:,0], sr, sr16k)
        save_wav(file_path_0, y0, sr=sr16k)
        file_path_1= change_suffix(file_path_in, file_out_16k, new_suffix="_16k_R")
        y1=librosa.resample(w[:,1], sr, sr16k)
        save_wav(file_path_1, y1, sr=sr16k)
        
