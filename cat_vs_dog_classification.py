import cv2 
import numpy as np
import os
from random import shuffle
from tqdm import tqdm 

TRAIN_DIR = 'directory to the train images dataset'
TEST_DIR = 'directory to the test images dataset'


IMG_SIZE = 50	# image size
LR = 1e-3 	# learning rate

# name under which we will save our mdel
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic-50pxls')



# this function will get the images name and return a vector corresponding to the image 
#([1,0] for cat and [0,1] for dog)
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    else: return [0,1]
    
    
# this function create a train data with the images in our train dataset(TRAIN_DIR) and return a 
#numpy array containing the features and labels(features are values of pixels of images)     
def create_train_data():
    training_data = []
    
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
        
    shuffle(training_data)
    np.save('training_data.npy', training_data)
    return training_data


# create test data with our test images in TEST_DIR by return a numpy array containing features and labels
# here the labels are just the images numbers.

def process_test_data():
    testing_data = []
    
    for img in tqdm(os.listdir(TEST_DIR)):
        im_num = img.split('.')[0]
        path = os.path.join(TEST_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), im_num])
        
    shuffle(testing_data)    
    np.save('testing_data.npy', testing_data )
    
    return testing_data
    
    
# executing the functions create above    
train_data = create_train_data()
test_data = process_test_data()



# import tflearn for building our model and visualling on tensorborad 

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

# here we defined 6 layered convolutional neural network, with a fully connected layer, and then the output layer

convnet = input_data(shape = [None, IMG_SIZE, IMG_SIZE, 1], name= 'input')

convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation = 'relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation = 'sigmoid')
convnet = regression(convnet, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')





# if we already run the model before and saved it we can just load it
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    
    
# let's split out training and testing data: 
#The training data is what we'll fit the neural network with, and the test data is what we're going
#to use to validate the results
#the testing data will only be used to test the accuracy of the network, not to train it.

train = train_data[:-500]
test = train_data[-500:]    
    


# separate my features and labels in the array   
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

#Now we fit for 5 epochs and save it  
model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)


# visualizing 12 images on testing data and see what the prediction will  be
import matplotlib.pyplot as plt


test_data = np.load('testing_data.npy')

fig = plt.figure()

for num, data in tqdm(enumerate(test_data[200:212])):
    img_num = data[1]
    img_data = data[0]
    
    
    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    model_out = model.predict(data)[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'   
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

