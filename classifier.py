from keras.models import *
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt
import numpy as np

x_train = np.load("train_feature.npy")
x_train = x_train.reshape((1000,2048))
y_train = np.load("train_label.npy")

x_test = np.load("test_feature.npy")
x_test = x_test.reshape((200,2048))
y_test = np.load("test_label.npy")

from keras.utils.np_utils import to_categorical
y_binary_train = to_categorical(y_train)
y_binary_test = to_categorical(y_test)

##################################################
model = Sequential()
model.add(Dense(units=32 , input_dim = 2048 ))
model.add(Dense(units=2))



model.compile(loss='mean_squared_error',
	          optimizer = 'adam',
	          metrics = ['accuracy'])

history = model.fit(x_train, y_binary_train, validation_data= (x_test, y_binary_test), epochs=10, batch_size=3)
##################################################



# list all data in history

print(history.history.keys())
# summarize history for accuracy while trainning
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss while trainning
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
print (model.test_on_batch(x_test, y_binary_test))