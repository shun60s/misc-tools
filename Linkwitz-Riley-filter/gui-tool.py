#coding:utf-8

#
# A GUI tool to display Linkwitz-Riley filter frequency response for design audio crossover filter. 
# by tkinter
#--------------------------------------------------------------
# This is based on <https://www.wizard-notes.com/entry/python/audio-crossover-filter-2-band> python code.
#--------------------------------------------------------------
#  Using 
# Python 3.10.4, 64bit on Win32 (Windows 10)
# numpy 1.21.6
# scipy 1.8.0
# mathplotlib 3.5.2
# --------------------
#  Using
# Python 3.10.12 (Ubuntu 22.04.4 LTS)
# numpy 2.1.0
# scipy 1.14.0
# matplotlib 3.9.2
#  
# It may need that sudo apt-get install python3-pil python3-pil.imagetk.
#  -----------

import os
import sys
import re
import argparse


import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter.font import Font

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


import scipy.signal
import numpy as np

class GUI_App(ttk.Frame):
    def __init__(self, master=None):
        self.frame=ttk.Frame.__init__(self,master)
        #
        self.enable_print = True
        #
        self.create_widgets()
        
        # rediect piint out to Text 
        sys.stdout= self.StdoutRedirector(self.text)
    
    
    
    def create_widgets(self,):
        #######################################################################################
        self.frame0=ttk.Frame(self.frame)
        self.frame0.pack(side=tk.LEFT)
        fig = plt.figure()
        self.canvas = FigureCanvasTkAgg(fig, self.frame0)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        
        
        #######################################################################################
        self.frame4=ttk.Frame(self.frame)
        
        # Entry label
        gyou=0
        self.label40= ttk.Label(self.frame4, text='crossover (cut off) [Hz]',width=25)
        self.entry40= ttk.Entry(self.frame4)
        self.entry40.insert(0,'1000')
        self.label40.grid(row=gyou, column=0)
        self.entry40.grid(row=gyou, column=1)
        
        gyou=gyou+1
        self.label41= ttk.Label(self.frame4, text='Butterworth filter order',width=25)
        self.entry41= ttk.Entry(self.frame4)
        self.entry41.insert(0,'2')
        self.label41.grid(row=gyou, column=0)
        self.entry41.grid(row=gyou, column=1)
        
        
        gyou=gyou+1
        self.label43= ttk.Label(self.frame4, text='sampling rate [Hz]',width=25)
        self.entry43= ttk.Entry(self.frame4)
        self.entry43.insert(0,'48000')
        self.label43.grid(row=gyou, column=0)
        self.entry43.grid(row=gyou, column=1)
        
        gyou=gyou+1
        self.buttonb1 = ttk.Button(self.frame4, text='display Linkwitz-Riley filter', width=29, command= self.button1_clicked)
        self.buttonb1.grid(row=gyou, column=0)
        
        gyou=gyou+1
        self.buttonb2 = ttk.Button(self.frame4, text='display Butterworth filter', width=29, command= self.button2_clicked)
        self.buttonb2.grid(row=gyou, column=0)
        
        self.frame4.pack(fill = X)      
        
        
        #######################################################################################
        self.frame7=ttk.Frame(self.frame)
        # text
        gyou=0
        f = Font(family='Helvetica', size=8)
        tv1 = StringVar()
        self.text = Text(self.frame7, height=3, width=45)
        self.text.configure(font=f)
        self.text.grid(row=gyou, column=0, sticky=(N, W, S, E))
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame7, orient=VERTICAL, command=self.text.yview)
        self.text['yscrollcommand'] = scrollbar.set
        scrollbar.grid(row=gyou, column=1, sticky=(N, S))
        
        
        self.frame7.pack(side=LEFT)
        #######################################################################################
        
        
    def button1_clicked(self,):
        #
        #self.button1['text']=('in process')
        #self.button1.configure(state=DISABLED)
        #
        self.amp1 = True
        if self.enable_print:
            #print ('button1 was clicked')
            print ('crossover (cut off) [Hz]', int(self.entry40.get()))
            print ('Butterworth filter order', int(self.entry41.get()))
            print ('sampling rate [Hz]', int(self.entry43.get()))
        
        self.cal_filter()
        self.make_draw(2)
        #canvas = FigureCanvasTkAgg(fig, self.frame0)
        self.canvas.draw()
        #canvas.get_tk_widget().pack()
        
    def button2_clicked(self,):
        #
        #self.button1['text']=('in process')
        #self.button1.configure(state=DISABLED)
        #
        self.amp1 = True
        if self.enable_print:
            print ('button2 was clicked')
            print ('crossover (cut off) [Hz]', int(self.entry40.get()))
            print ('Butterworth filter order', int(self.entry41.get()))
            print ('sampling rate [Hz]', int(self.entry43.get()))
        
        self.cal_filter()
        self.make_draw(1)
        #canvas = FigureCanvasTkAgg(fig, self.frame0)
        self.canvas.draw()
        #canvas.get_tk_widget().pack()
        
    def cal_filter(self,):
        # バターワースフィルタのフィルタ係数算出
        nfc = int(self.entry40.get())  #　カットオフ周波数[Hz]
        nfs= int(self.entry43.get()) #　サンプリング周波数 [Hz]
        self.order = int(self.entry41.get()) #　フィルタ次数
        
        nanalog=False  #filter type is "digital filter"
        
        b, a = scipy.signal.butter(self.order, nfc, btype='low', analog=nanalog, fs=nfs)
        if self.enable_print:
            print('low filter coefficients:')
            print(b, a)
        self.w, self.h_bw_low  = scipy.signal.freqz(b, a, fs=nfs)
        b, a = scipy.signal.butter(self.order, nfc, btype='high', analog=nanalog, fs=nfs)
        if self.enable_print:
            print('high filter coefficients:')
            print(b, a)
        self.w, self.h_bw_high = scipy.signal.freqz(b, a, fs=nfs)
        
        # Linkwitz-Riley　フィルタの応答算出
        self.h_lr_low  = self.h_bw_low**2
        self.h_lr_high = self.h_bw_high**2
        # 2, 6, ... 次Linkwitz-Riley　フィルタの場合、片方の極性を反転
        self.h_lr_high = -self.h_lr_high if (self.order*2+2) % 4 == 0 else self.h_lr_high
    
    def plot_mag(self, w, h, label):
        # 振幅特性プロット用
        if type(h) != list:
            sum_h = h
        else:
            sum_h = 0.0
            for tmp_h in h:
                sum_h = sum_h + tmp_h
        plt.xscale('log')
        plt.plot(w, 20 * np.log10(np.abs(sum_h)+ 10e-20), label=label, alpha=0.5)
    
    def plot_phase(self, w, h, label):
        # 位相特性プロット用
        if type(h) != list:
            sum_h = h
        else:
            sum_h = 0.0
            for tmp_h in h:
                sum_h = sum_h + tmp_h
        plt.xscale('log')
        plt.plot(w, np.angle(sum_h) * 180 / np.pi, label=label, alpha=0.5)
        
    def make_draw(self,kind):
        #
        #fig = plt.figure()
        
        if kind == 2:
            self.low=self.h_lr_low
            self.high=self.h_lr_high
            self.filter='Linkwitz-Riley'
        elif kind == 1:
            self.low=self.h_bw_low
            self.high=self.h_bw_high
            self.filter='Butterworth'
        
        if 1:
            self.cal_filter()
            
            plt.subplot(2,1,1)
            plt.cla()
            self.plot_mag(self.w, self.low, f"{self.order * kind}-{self.filter} low")
            self.plot_mag(self.w, self.high, f"{self.order * kind}-{self.filter} high")
            self.plot_mag(self.w, [self.low, self.high], f"{self.order * kind}-{self.filter} sum")
            
            plt.ylabel('Magnitude [dB]')
            plt.xlabel('Frequency [Hz]')
            plt.ylim(-25, 5)
            plt.grid(which='both', axis='both')
            plt.legend(loc='lower left')
            
            plt.subplot(2,1,2)
            plt.cla()
            self.plot_phase(self.w, self.low, f"{self.order * kind}-{self.filter} low")
            self.plot_phase(self.w, self.high, f"{self.order * kind}-{self.filter} high")
            self.plot_phase(self.w, [self.low, self.high], f"{self.order * kind}-{self.filter} sum")
            
            plt.ylabel('Phase (deg)')
            plt.xlabel('Frequency [Hz]')
            plt.grid(which='both', axis='both')
            plt.legend(loc='lower left')
            plt.ylim(-180, 180)
            
            
        
        plt.tight_layout()
        
        
    
    
    
    class IORedirector(object):
        def __init__(self, text_area):
            self.text_area = text_area
            self.line_flag = False
            
    class StdoutRedirector(IORedirector):
        def write(self,st):
            if 0: # esc-r
                # check if there is num/num in st
                if re.search(r' \d+/\d+',st) is not None:
                    #
                    if not self.line_flag:  # start...
                        self.text_area.insert('end',  st)
                        self.text_area.insert('end', "\n") # make index up
                        self.line_flag=True
                    else:
                        # delete last 1 line 
                        self.text_area.delete("end-2l", "end-1l")
                        #
                        self.text_area.insert('end', st)
                        self.text_area.insert('end', "\n") # make index up
                else:
                    #self.text_area.insert('end', pos +'>' + st)
                    self.text_area.insert('end',  st )
                    if st != "":
                        self.line_flag = False  # reset line_flag
            else:
                self.text_area.insert('end',  st)
            
            self.text_area.see("end")
        
        
        def flush(self):
            pass




def quit_1(root_window):
    root_window.quit()
    root_window.destroy()


sys.stdout= sys.__stdout__


if __name__ == '__main__':
    #
    root = Tk()
    root.protocol("WM_DELETE_WINDOW", lambda :quit_1(root))
    root.title('display frequency response')
    
    app=GUI_App(master=root)
    app.mainloop()
