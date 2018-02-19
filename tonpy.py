import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np
a = np.load('train_feature.npy')
a = a.reshape((1000,2048))
b = np.load('test_label.npy')
print b.shape
print b

'''
aNew = a.resize((1000,2048))
bNew = b.resize((200,2048))

np.save('train_feature.npy',aNew)
np.save('test_feature.npy',bNew)
'''
