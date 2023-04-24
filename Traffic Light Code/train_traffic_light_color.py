# Project: How to Detect and Classify Traffic Lights
# Author: Addison Sears-Collins
 
import collections 
import cv2
import matplotlib.pyplot as plt 
import numpy as np 
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import glob 

# To detect objects, we will use a pretrained neural network that has been 
# COCO labels are here: https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
 
 
def get_files(pattern):
  files = []
  for file_name in glob.iglob(pattern, recursive=True):
    files.append(file_name)
  return files
 
def load_rgb_images(pattern, shape=None):
  files = get_files(pattern)
  images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]
  if shape:
    return [cv2.resize(img, shape) for img in images]
  else:
    return images
 
def center(box, coord_type):
  return (box[coord_type] + box[coord_type + "2"]) / 2

def double_shuffle(images, labels):
  indexes = np.random.permutation(len(images))
  return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]
 
def reverse_preprocess_inception(img_preprocessed):
  img = img_preprocessed + 1.0
  img = img * 127.5
  return img.astype(np.uint8)

sys.path.append('../')
 
# Show the version of TensorFlow and Keras that I am using
print("TensorFlow", tf.__version__)
print("Keras", keras.__version__)
 
def show_history(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
  plt.show()
 
def Transfer(n_classes, freeze_layers=True):
  print("Loading Inception V3...")
  # input_shape needs to have 3 channels, and needs to be at least 75x75 for the
  # resolution.
  # Our neural network will build off of the Inception V3 model (trained on the ImageNet
  # data set).
  base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
 
  print("Inception V3 has finished loading.")
  
  print('Layers: ', len(base_model.layers))
  print("Shape:", base_model.output_shape[1:])
  print("Shape:", base_model.output_shape)
  print("Shape:", base_model.outputs)
  base_model.summary()
 
  # Create the neural network. This network uses the Sequential
  # architecture where each layer has one 
  # input tensor (e.g. vector, matrix, etc.) and one output tensor 
  top_model = Sequential()
 
  # Our classifier model will build on top of the base model
  top_model.add(base_model)
  
  top_model.add(GlobalAveragePooling2D())
  top_model.add(Dropout(0.5))
  top_model.add(Dense(1024, activation='relu'))
  top_model.add(BatchNormalization())
  top_model.add(Dropout(0.5))
  top_model.add(Dense(512, activation='relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(128, activation='relu'))
  top_model.add(Dense(n_classes, activation='softmax'))
 
  if freeze_layers:
    for layer in base_model.layers:
      layer.trainable = False
 
  return top_model
 
# data augmentation
datagen = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             zoom_range=[0.7, 1.5], height_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             horizontal_flip=True)
 
shape = (299, 299)
 
img_0_green = load_rgb_images("./traffic_light_dataset/0_green/*", shape)
img_1_yellow = load_rgb_images("./traffic_light_dataset/1_yellow/*", shape)
img_2_red = load_rgb_images("./traffic_light_dataset/2_red/*", shape)
img_3_not_traffic_light = load_rgb_images("traffic_light_dataset/3_not/*", shape)
 
# Create a list of the labels that is the same length as the number of images in each category
# 0 = green
# 1 = yellow
# 2 = red
# 3 = not a traffic light
labels = [0] * len(img_0_green)
labels.extend([1] * len(img_1_yellow))
labels.extend([2] * len(img_2_red))
labels.extend([3] * len(img_3_not_traffic_light))
 
# Create NumPy array
labels_np = np.ndarray(shape=(len(labels), 4))
images_np = np.ndarray(shape=(len(labels), shape[0], shape[1], 3))
 
# Create a list of all the images in the traffic lights data set
img_all = []
img_all.extend(img_0_green)
img_all.extend(img_1_yellow)
img_all.extend(img_2_red)
img_all.extend(img_3_not_traffic_light)
 
# Make sure we have the same number of images as we have labels
assert len(img_all) == len(labels)  
 
# Shuffle the images
img_all = [preprocess_input(img) for img in img_all]
(img_all, labels) = double_shuffle(img_all, labels)
 
# Store images and labels in a NumPy array
for idx in range(len(labels)):
  images_np[idx] = img_all[idx]
  labels_np[idx] = labels[idx]
     
print("Images: ", len(img_all))
print("Labels: ", len(labels))
 
# Perform one-hot encoding
for idx in range(len(labels_np)):
  labels_np[idx] = np.array(to_categorical(labels[idx], 4))
  # Split - %80 training %20 validation
idx_split = int(len(labels_np) * 0.8)
x_train = images_np[0:idx_split]
x_valid = images_np[idx_split:]
y_train = labels_np[0:idx_split]
y_valid = labels_np[idx_split:]
 
# Store a count of the number of traffic lights of each color
cnt = collections.Counter(labels)
print('Labels:', cnt)
n = len(labels)
print('0:', cnt[0])
print('1:', cnt[1])
print('2:', cnt[2])
print('3:', cnt[3])
 
# Calculate the weighting of each traffic light class
class_weight = {0: n / cnt[0], 1: n / cnt[1], 2: n / cnt[2], 3: n / cnt[3]}
print('Class weight:', class_weight)
 
# Save the best model as traffic.h5
checkpoint = ModelCheckpoint("traffic.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.0005, patience=15, verbose=1)

model = Transfer(n_classes=4, freeze_layers=True)

model.summary()
 
it_train = datagen.flow(x_train, y_train, batch_size=32)
 
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(
  learning_rate=1.0, rho=0.95, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
 
# Train the model on the image batches for a fixed number of epochs
# Store a record of the error on the training data set and metrics values
#   in the history object.
history_object = model.fit(it_train, epochs=250, validation_data=(
  x_valid, y_valid), shuffle=True, callbacks=[
  checkpoint, early_stopping], class_weight=class_weight)
 
# Display the training history
show_history(history_object)
 
# Get the loss value and metrics values on the validation data set
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
 
print('Saving the validation data set...')
 
print('Length of the validation data set:', len(x_valid))
 
