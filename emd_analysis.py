#coding:utf-8

#  音の開始時点を推定？するために作ったもの。
#  spectrogramにおいて隣接する変化の大きな場所を見つける
#  EMD(Empirical Mode Decomposition)で IMF波形を求め 目標に一番近いIMF番号を探して波形を観察する



import sys
import os
import glob
import argparse
import pathlib
import math


import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write as wavwrite
from scipy import signal # version is 1.4.1
from scipy.signal import hilbert

from matplotlib import pyplot as plt # version is 3.3.1
import matplotlib.colors as colors

import emd  # version emd should 0.3.3 by this program 


def read_wav( file_path ):
    try:
        sr, w = wavread( file_path)
    except:
        print ('error: wavread ', file_path)
        sys.exit()
    else:
        w= w / (2 ** 15)
        print ('sampling rate ', sr)
        print ('size', w.shape)
    return w, sr

def save_wav( file_path, data, sr=48000):
    amplitude = np.iinfo(np.int16).max
    try:
        wavwrite( file_path , sr, np.array( amplitude * data , dtype=np.int16))
    except:
        print ('error: wavwrite ', file_path)
        sys.exit()
    print ('wrote ', file_path)




# ピーク最小幅
MIN_WIDTH= 9    # 最小の時間幅　ｍＳ単位
MIN_HIGH_FOR_TEST = 0.5   # ピークの最小高さ test信号用

def spec_diff_x(spec, bins=None, MIN_HIGH=MIN_HIGH_FOR_TEST, soukan=False):
    iy=spec.shape[1]  # time
    ix=spec.shape[0]  # 周波数軸
    
    
    if soukan:  # 相関係数
        spec_diff_mean=np.zeros(iy-1)
        for i in range( iy-1):
            ce=np.corrcoef(  spec[:,i], spec[:,i+1] )
            #print ('ce', ce)
            spec_diff_mean[i]= ce[0,1]
        
    else:
        spec_diff= np.abs(spec[:,1:iy] - spec[:,0:iy-1])  # 差の絶対値
        spec_diff_mean= np.mean( spec_diff,axis=0)        # 周波数軸に沿って平均値
    

    
    peaks, _ = signal.find_peaks(spec_diff_mean, height= MIN_HIGH * max(spec_diff_mean) )
    
    ix= np.argmax(spec_diff_mean[peaks])
    if bins is not None:
        t_point= bins[peaks[ix]+1] # 変化後のためプラス１する
        return spec_diff_mean, peaks, t_point
    else:
        return spec_diff_mean, peaks # 差をとるので次数が1個減る


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='find start point')
    parser.add_argument('--dir', '-d', default='WAV_MONO/', help='specify wav directory')
    args = parser.parse_args()
    
    if 0:
        flist= \
        [
         "WAV_MONO/1.wav",
        ]
    else:
        flist= glob.glob( args.dir + '*.wav')
    
    # スペクトログラムの計算のためのパラメーターの設定
    NFFT=256    # 1スパンサイズ
    NOVERLAP=int(NFFT * 0.8)  # オーバーラップサイズ量（シフト量とは違う）
    YLIMH=5000  # 表示の上限周波数
    YLIML=100   # 表示の下限周波数
    YLIMH2=5000  # 比較のための上限周波数
    YLIML2=100   # 比較のための下限周波数
    
    # 波形表示のためのパラメーターの設定
    MIN_SPAN=  100  # 最小の時間幅　ｍＳ単位
    
    #spectrogram間の差の比較のための設定
    MIN_SPEC_SA= 10
    
    for file_path in flist:
        # wavを読み込む
        w_ref,sr= read_wav( file_path)
        print (file_path, sr)
        #
        fig = plt.figure()
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)
        
        # spectrogramを計算する
        Pxxl, freqsl, binsl, iml = ax1.specgram(w_ref, NFFT=NFFT, Fs=sr, noverlap=NOVERLAP)
        low_index= np.amin(np.where( freqsl > YLIML2))
        high_index= np.amax(np.where( freqsl < YLIMH2))
        print('index', low_index, high_index, freqsl[low_index], freqsl[high_index])
        if 1:
            fig.colorbar(iml, ax=ax1).set_label('Intensity [dB]')
        
        ax1.grid(which='both', axis='both')
        ax1.set_ylim(YLIML,YLIMH)
        specl= np.log10(Pxxl[low_index:high_index,:]) *10
        
        # spectrogram間の相関係数の計算
        spec_soukan, peaksw, t_pointw = spec_diff_x(specl,binsl,soukan=True)
        ax2.plot(binsl[:-1], spec_soukan)
        ax2.set_xlim(binsl[0],binsl[-1])
        
        # spectrogram間の差の計算
        spec_diffw, peaksw, t_pointw = spec_diff_x(specl,binsl,soukan=False)
        ax3.plot(binsl[:-1], spec_diffw)
        ax3.set_xlim(binsl[0],binsl[-1])
        print ('t_pointw', t_pointw)
        # wav 表示
        x_time= np.linspace(0, len( w_ref)/sr, len(w_ref))
        ax4.plot(x_time, w_ref)
        ax4.set_xlim( max(0,t_pointw - (MIN_SPAN/2)/1000), min(x_time[-1],t_pointw + (MIN_SPAN/2)/1000))
        ax4.set_title(str(t_pointw))
        ax4.plot(t_pointw, w_ref[ int(t_pointw * sr)], "o")
        plt.tight_layout()
        
        #  ここでt_pointw 以降の波形を切り出す
        st0= t_pointw  #0
        ep0=len(w_ref)/sr
        w_ref2= w_ref[ int(sr*st0): int(sr*ep0)]
        imf = emd.sift.sift(w_ref2)
        print(imf.shape)
        imf_max_index=5
        # IMFの取り出すインデックスの数を指定する
        imf2=imf[:,0: min(imf_max_index, imf.shape[1])]
        emd.plotting.plot_imfs(imf2, scale_y=True, cmap=True)
        
        IP, IF, IA = emd.spectra.frequency_transform(imf2, sr,  'nht') #'hilbert')
        # IP : ndarray
        # Array of instantaneous phase estimates
        # IF : ndarray
        # Array of instantaneous frequency estimates
        # IA : ndarray
        # Array of instantaneous amplitude estimates
        
        freq_edges, freq_bins = emd.spectra.define_hist_bins(1, 5000, 100)  #, scale='log')
        hht = emd.spectra.hilberthuang(IF, IA, freq_edges)
        print ('hht.shape', hht.shape)   # bands , IMF size
        print ('imf size', len(imf2), len(w_ref2))
        print ('IF.shape', IF.shape)
        
        # ピックアップするターゲット周波数に一番近いIMFを探す
        TGT_FREQ=1000
        index_tg= np.argmin((np.abs(np.mean( IF, axis=0) - TGT_FREQ)))
        print ('IF mean', np.mean( IF, axis=0))
        print ('nearest IF', np.mean( IF, axis=0)[index_tg], index_tg)
        
        # パワー表示？　パワーの大小が大きすぎて表示が上手く行っていない
        x_time2= np.linspace(0, len( w_ref2)/sr, len(w_ref2))
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.pcolormesh(x_time2[:len(imf2)], freq_bins, hht[:, :len(imf2)], cmap='Reds', shading='auto', norm=colors.LogNorm(vmin=1e-7))
        cb = plt.colorbar()
        cb.set_label('Power')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (secs)')
        
        plt.show()
        plt.close()
        
