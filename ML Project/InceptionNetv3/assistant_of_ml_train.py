# huggingface-hub==0.28.1
# torch==2.6.0
# tensorflow==2.18.0
# pandas==2.2.3
# matplotlib==3.10.0
# seaborn==0.13.2
# scikit-learn==1.6.1
#importing gpu
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
config = ConfigProto()
from datetime import datetime
import time
from pathlib import Path
import os
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

class assistant_for_model_training:
    def __init__(self):
        pass

    def check_dependencies(self,config,tf,InteractiveSession):
        #checking gpu function availability
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        #session-configuration adjusting
        session = InteractiveSession(config=config)
        #checking tensorflow-gpu support
        print(tf.__version__)
        print(len(tf.config.list_physical_devices('GPU'))>0)
        print('Tensorflow Version: ', tf.__version__)
        print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
        print('CUDA Version', tf.sysconfig.get_build_info()['cuda_version'])
        print('CUDNN Version', tf.sysconfig.get_build_info()['cudnn_version'])

    def save_dataset(self,df, dir_path, data_name):
        data_dir = os.path.join(dir_path, data_name)
        os.makedirs(data_dir, exist_ok=True)

        # Iterate through the dataframe and save images in the corresponding label subdirectories
        for _, row in df.iterrows():
            label = row['label']
            filepath = row['filepath']
            label_dir = os.path.join(data_dir, label)

            # Ensure the label directory exists
            os.makedirs(label_dir, exist_ok=True)

            # Copy the image to the appropriate label directory
            try:
                shutil.copy(filepath, os.path.join(label_dir, os.path.basename(filepath)))
            except Exception as e:
                print(f"Error saving image {os.path.basename(filepath)}: {e}")
        print(f"Dataset saved in {data_dir}")

    #Count images in directory
    def count_files_in_dir(self,dir_path: str):
        file_counts = {}
        for path in Path(dir_path).rglob('*'):
            if path.is_file():
                dir_path = path.parent
                if dir_path in file_counts:
                    file_counts[dir_path] += 1
                else:
                    file_counts[dir_path] = 1
        return file_counts
    
    #Display Directory name and number of files
    def display_dir_file_count(self,dir_path: str):
        file_counts = self.count_files_in_dir(dir_path)
        for dir_path, count in file_counts.items():
            print(f"Directory: {dir_path} has {count} files")

    #Create dataframe from dataset
    def create_dataframe_from_dataset(self,IMAGE_EXTENSIONS,dir_path: str):
        # label_encoder = LabelEncoder()
        filepaths = []
        labels = []

        for label in os.listdir(dir_path):
            label_dir = os.path.join(dir_path, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith(IMAGE_EXTENSIONS):
                        filepaths.append(os.path.join(label_dir, file))
                        labels.append(label)
        df = pd.DataFrame({
            'filepath': filepaths,
            'label': labels
        })
        # df['label_encoded'] = label_encoder.fit_transform(df['label'])
        return df

    def get_number_of_classes(self,dir_path):
        return len(os.listdir(dir_path))
    
    def train_test_generator(self):
        train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],)
        test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input )
        return train_generator,test_generator
    
    def train_validation_test_set(self,train_generator,train_df,test_generator,test_df):
        # Split the data into three categories.
        training_set = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='filepath',
            y_col='label',
            target_size=(299, 299),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=64,
            shuffle=True,
            seed=69,
            subset='training'
        )

        validation_set = train_generator.flow_from_dataframe(
            dataframe=train_df,
            x_col='filepath',
            y_col='label',
            target_size=(299, 299),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=64,
            shuffle=True,
            seed=69,
            subset='validation'
        )

        test_set = test_generator.flow_from_dataframe(
            dataframe=test_df,
            x_col='filepath',
            y_col='label',
            target_size=(299, 299),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=64,
            shuffle=False
        )
        return training_set,validation_set,test_set
    
    def get_pretrained_model(self,IMAGE_SIZE):
        #Pretrained model structure fetching with custom dataset
        inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
        inception.trainable = True
        print("Number of layers in the base model: ", len(inception.layers))
        # Fine-tune from this layer onwards
        fine_tune_at = 180
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in inception.layers[:fine_tune_at]:
            layer.trainable = False
        print("The number of trainable layers: {}".format(len(inception.layers)-fine_tune_at))
        print("The number of untrainable layers: {}".format(fine_tune_at))
        return inception,fine_tune_at
    
    def get_model_performance_plot(self, trained_model, save_path_loss='loss_plot.png', save_path_acc='acc_plot.png'):
        # Model performance plotting for loss
        plt.plot(trained_model.history['loss'], label='train loss')
        plt.plot(trained_model.history['val_loss'], label='val loss')
        plt.legend()
        plt.savefig(save_path_loss)  # Save the loss plot as an image
        plt.close()  # Close the plot to avoid overlapping with the next plot
        
        # Model performance plotting for accuracy
        plt.plot(trained_model.history['accuracy'], label='train acc')
        plt.plot(trained_model.history['val_accuracy'], label='val acc')
        plt.legend()
        plt.savefig(save_path_acc)  # Save the accuracy plot as an image
        plt.close()  # Close the plot

    def get_confusion_matrix_of_during_trained(self,model,test_set):
        #Model performance on validation dataset
        y_pred = model.predict(test_set)
        #Calling best match array
        y_pred = np.argmax(y_pred, axis=1)
        # Assuming y_pred contains probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)
        # Use y_pred_classes for confusion matrix and accuracy score
        truth_chart = confusion_matrix(test_set.classes, y_pred_classes)
        print(truth_chart)
        accuracy = accuracy_score(test_set.classes, y_pred_classes)
        print(accuracy)
        return truth_chart,accuracy
    
    def get_confusion_matrix(self, model, test_set):
        # Model performance on validation dataset
        y_pred = model.predict(test_set)
        
        # Assuming y_pred contains probabilities/logits
        y_pred_classes = np.argmax(y_pred, axis=1)  # Take the class with the highest probability
        
        # Use y_pred_classes for confusion matrix and accuracy score
        truth_chart = confusion_matrix(test_set.classes, y_pred_classes)
        print("Confusion Matrix:\n", truth_chart)
        
        accuracy = accuracy_score(test_set.classes, y_pred_classes)
        print("Accuracy:", accuracy)
        
        return truth_chart, accuracy
        
    def save_the_trained_model(self,model,accuracy,fine_tune_at):
        # current timestamp
        dt = datetime.now()
        x= datetime.timestamp(dt)
        # convert to datetime
        date_time = datetime.fromtimestamp(x)
        str_time = date_time.strftime("%H%M_%d%B%Y")
        acc =round(accuracy*100,3)
        file_name ='mustard_inception_Fine_tuned_'+str(fine_tune_at)+'_acc' +str(acc)+'_'+str_time+'.h5'
        loc = file_name
        model.save(loc)
        return file_name

    def get_the_model(self,model_path): 
        model = load_model(model_path)
        return model


