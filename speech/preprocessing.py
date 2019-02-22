# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:30:48 2018

@author: Александр
"""


import numpy as np
#from pydub import AudioSegment
from scipy.io import wavfile 
from math import log

#AudioSegment.ffmpeg = "D:/ffmpeg/bin"
filename = "f3_16k_003"
samplingFrequency, signalData = wavfile.read(filename+'.wav') 

num = len(signalData) // (samplingFrequency / 2)
f1 = np.pad(signalData, ((0,int((num+1)*samplingFrequency / 2-len(signalData))),(0,0)),'constant', constant_values = 0)
f1 = np.concatenate([f1[:,0],f1[:,1]])

f1 = list(f1)
infile = []
del signalData
snum = 0
count = 0
for i in f1:
    k = log(1+255*abs(i/32768.0), 256)
    if i < 0:
        k = -k
    infile.append(k)
    if (len(infile)==12000*samplingFrequency/2) or (count == len(f1)-1):
        infile = np.reshape(infile, (-1, int(samplingFrequency/2)))
        np.save('inae/'+filename+'_'+str(snum), infile)
        #np.save('f3_in1', infile)
        snum+=1
        print(snum)
        del infile
        infile = []
    count +=1
#infile = np.reshape(infile, (-1, int(samplingFrequency/2)))
#np.save("f2_16k",infile)
    
del infile, f1