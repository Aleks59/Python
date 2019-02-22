import numpy as np
from math import pow
from scipy.io.wavfile import write

file_in = np.load("cvae_out.npy")

#f1 = np.concatenate([signalData[:,0],signalData[:,1]])
#num = len(f1) // (samplingFrequency / 2)
#f1 = list(f1[int(len(f1) - num * samplingFrequency / 2) : ])
#infile = []
#del signalData
#for i in f1:
#    i = log(1+255*abs(i/32768.0), 256)
#    if i < 0:
#        i = -i
#    infile.append(i)
#infile = np.reshape(infile, (-1, int(samplingFrequency/2)))
#np.save("eval_in",infile)

file_in = np.reshape(file_in, (-1))   
f1 = []
for i in file_in:
    k = int((pow(256,abs(i))-1)/255*32768)
    if i<0:
        k=-k
    f1.append(k)
f1 = np.array(f1, dtype = "int16")
f1 = np.reshape(f1, (2,-1))
write("cvae_out.wav", 16000, f1.T)