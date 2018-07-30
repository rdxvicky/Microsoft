import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten, MaxPooling2D, Reshape, InputLayer


# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


root_dir = os.path.abspath('../..')
print(root_dir)
data_dir = os.path.join(root_dir, 'tensorflow/medical')
sub_dir = os.path.join(root_dir, 'Downloads')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
os.path.exists(sub_dir)


train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test.csv'))

sample_submission = pd.read_csv(os.path.join(data_dir, 'Sample_Submission.csv'))

temp = []
for img_name in train.id:
  image_path = os.path.join(data_dir, 'Train', 'images', 'train', str(img_name)+'.png')
  img = imread(image_path, flatten=True)
  img = img.astype('float32')
  temp.append(img)

train_x = np.stack(temp)

#train_x /= 255.0
train_x = train_x.reshape(train_x.shape[0], 64, 64, 1).astype('float32')
train_x /= 255.0

temp = []
for img_name in test.id:
  image_path = os.path.join(data_dir, 'Train', 'images', 'test', str(img_name)+'.png')
  img = imread(image_path, flatten=True)
  img = img.astype('float32')
  temp.append(img)

test_x = np.stack(temp)


test_x = test_x.reshape(test_x.shape[0], 64, 64, 1).astype('float32')
test_x /= 255.0

train_y = keras.utils.np_utils.to_categorical(train.orientation.values)


split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]


# reshape data

train_x_temp = train_x.reshape(train_x.shape[0], 64, 64, 1)
val_x_temp = val_x.reshape(val_x.shape[0], 64, 64, 1)


# define vars
input_shape = (4096,)
input_reshape = (64, 64, 1)

conv_num_filters = 5
conv_filter_size = 5

pool_size = (2, 2)

hidden_num_units = 100
output_num_units = 4

epochs = 5
batch_size = 128

model = Sequential([
 InputLayer(input_shape=input_reshape),

 Convolution2D(64, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(64, 5, 5, activation='relu'),
 MaxPooling2D(pool_size=pool_size),

 Convolution2D(64, 4, 4, activation='relu'),

 Flatten(),

 Dense(output_dim=hidden_num_units, activation='relu'),

 Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model_conv = model.fit(train_x_temp, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x_temp, val_y))

pred = model.predict_classes(test_x)

sample_submission.id = test.id; sample_submission.orientation = pred
sample_submission.to_csv(os.path.join(sub_dir, 'sub03.csv'), index=False)
