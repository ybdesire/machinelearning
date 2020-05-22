from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt


[Fs, x] = audioBasicIO.readAudioFile("happy.wav");
# Fs is frequency
# x is real data


F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
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




'''
Fs=16000 # audio frequency = 16000
x.shape=(23776,) # audio length = 23776
F.shape=(34, 58) # 34 kinds of features, each feature has 58 length vector
f_names len=34 # 34 features below
f_names=['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy', 'spectral_flux',
 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10'
, 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', '
chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'chroma_std']
'''


print('Fs={0}'.format(Fs))
print('x.shape={0}'.format(x.shape))
print('F.shape={0}'.format(F.shape))
print('f_names len={0}'.format(len(f_names)))
print('f_names={0}'.format( f_names ))


plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]); 
plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
