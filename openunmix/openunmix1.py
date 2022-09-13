#coding:utf-8

# Open-Unmix provides ready-to-use models that allow users to separate pop music into four stems:
# vocals(0), drums(1), bass(2) and the remaining other instruments(3)

import torch
import torchaudio
import soundfile as sf

# Check version
# Python 3.6.4 on win32 (Windows 10)
# torch 1.7.1+cpu
# torchaudio 0.7.2
# soundfile 0.10.3.post1

# loading umxhq four target separator
separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq')
print ('separator.sample_rate', separator.sample_rate.detach().numpy())

SAMPLE_WAV_PATH="sample.wav"  # 44100 stereo

audio, sample_rate = torchaudio.load(filepath=SAMPLE_WAV_PATH)
if int(separator.sample_rate.detach().numpy()) is not sample_rate:
    print ('error: sample rate is missmatch')

# add nb_samples axis
x= torch.reshape(audio,(1,2,-1))

estimates = separator(x)

# remove nb_samples axis
y= torch.reshape(estimates,(4,2,-1))

# save separate audio track as wav
# vocals(0), drums(1), bass(2) and the remaining other instruments(3)
listx=['vocals','drumes','bass','others']
for i in range(4):
    y0= y[i]
    y0n= y0.detach().numpy() # trans tensor to numpy
    fn= listx[i] + '.wav'
    sf.write(fn, y0n.T, sample_rate)
    print (fn)


#torchaudio.save(filepath='out0.wav', src=y0, sample_rate=sample_rate) # not work well.



