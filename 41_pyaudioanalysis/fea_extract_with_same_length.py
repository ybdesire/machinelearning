from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt


[Fs1, x1] = audioBasicIO.readAudioFile("happy.wav");
[Fs2, x2] = audioBasicIO.readAudioFile("sad.wav");
# Fs is frequency
# x is real data

th = 100 # fixed fea length
k12 = (len(x1)-800)/th/float(Fs1)
k22 = (len(x2)-800)/th/float(Fs2)


F1, f_names1 = audioFeatureExtraction.stFeatureExtraction(x1, Fs1, 0.05*Fs1, k12*Fs1);
F2, f_names2 = audioFeatureExtraction.stFeatureExtraction(x2, Fs2, 0.05*Fs2, k22*Fs2);
# stFeatureExtraction(signal, fs, win, step):
# signal:       the input signal samples
# fs:           the sampling freq (in Hz)
# win:          the short-term window size (in samples)
# step:         the short-term window step (in samples)
'''
here, 
window size = 0.05*Fs = 0.05*16000 = 800
step size = 0.025*Fs = 0.024*16000 = 400
we can get n frames from signal with length 23776

400*n+800=23776 -> n=57.44 = 58

as below F.shape = (34,58)
'''




print('F1.shape={0}'.format(F1.shape))
print('F2.shape={0}'.format(F2.shape))


