#importing gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
config = ConfigProto()
from tensorflow.keras.models import load_model
from datetime import datetime
import time
from pathlib import Path
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import shutil
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from assistant_of_ml_train import assistant_for_model_training


assist_obj = assistant_for_model_training()  # create class instance
# assist_obj.check_dependencies(config,tf,InteractiveSession)


#calling global varibles
IMAGE_SIZE = [299, 299]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG')
dataset = "/home/gfl-ml-2025/Music/plant_not_plant/dataset/mustard_dataset"
batch_size = 64


# Walk through each directory
assist_obj.display_dir_file_count(dataset)
image_df = assist_obj.create_dataframe_from_dataset(IMAGE_EXTENSIONS,dataset)
# Separate in train and test data
train_df, test_df = train_test_split(image_df, test_size=0.1, shuffle=True, random_state=69)
assist_obj.save_dataset(test_df, os.getcwd(), 'test')
class_num = assist_obj.get_number_of_classes("/home/gfl-ml-2025/Music/plant_not_plant/test")
print("class_num : ",class_num)
train_generator,test_generator = assist_obj.train_test_generator()
training_set,validation_set,test_set = assist_obj.train_validation_test_set(train_generator,train_df,test_generator,test_df)

inception,fine_tune_at = assist_obj.get_pretrained_model(IMAGE_SIZE)
#Calling layers
x = Flatten()(inception.output)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(32, activation="relu")(x)
output = Dense(class_num, activation="softmax")(x)


model__ = load_model("core_model_inceptionv3_check_point.keras")
truth_chart,accuracy = assist_obj.get_confusion_matrix(model__,test_set)
model_path = assist_obj.save_the_trained_model(model__,accuracy,fine_tune_at)
model = assist_obj.get_the_model(model_path)
#Model evaluation on test_set
test_loss, test_accuracy = model.evaluate(test_set)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

