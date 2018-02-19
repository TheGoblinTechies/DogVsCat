from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import numpy as np
import cv2

def make_label(file_name):
    label = file_name.split('.')[0]
    ##one-hot-encoding
    if label == 'cat': 
        return [0]
    elif label == 'dog':
        return [1]


def make_data(img_path,img_size):
    path_length = len(os.listdir(img_path))
    print (path_length)
    images = np.zeros((path_length,224,224,3),dtype=np.uint8)
    labels = np.zeros((path_length,1),dtype=np.float32)
    count = 0
    for file_name in os.listdir(img_path):
        labels[count] = make_label(file_name)
        images[count] = cv2.resize(cv2.imread(img_path+'/'+file_name),(img_size,img_size))
        b,g,r = cv2.split(images[count])       # get b,g,r
        images[count] = cv2.merge([r,g,b])  # switch it to rgb
        count+=1
    ##shuffle
    p = np.random.permutation(path_length)
    images,labels = images[p],labels[p]
    print (labels.shape)
    return images,labels

x_train, y_train = make_data("train",224)
x_test , y_test  = make_data("test" ,224)

x = Input((224,224,3))
model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)


train = model.predict(x_train)
test = model.predict(x_test)
print (train.shape)
np.save("train_feature.npy",train)
np.save("test_feature.npy",test)
np.save("train_label.npy",y_train)
np.save("test_label.npy",y_test)
