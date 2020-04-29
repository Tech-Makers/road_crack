# Importing the libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications import VGG16

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.models import load_model
from keras.preprocessing import image
import unittest

class PictureManager(object):
          
    def clusterImage(self):
        classifier = load_model('Vggcd.h5') # loading the model
        print("Vggcd.h5 loaded")
        
        Crack_dict = {"[0]": "crack", 
                          "[1]": "non_crack"
                         }
        
        Crack_dict_n = {
                          "crack": "crack", 
                          "non_crack": "non_crack"
                          }
                         
        
        def draw_test(name, pred, im):
            
            cracking = Crack_dict[str(pred)]
            BLACK = [0,0,0]
            expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
            cv2.putText(expanded_image, cracking, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
            cv2.imshow(name, expanded_image)
        
        def getRandomImage(path):
            
            folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
            random_directory = np.random.randint(0,len(folders))
            path_class = folders[random_directory]
            print("Class - " + Crack_dict_n[str(path_class)])
            file_path = path + path_class
            file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
            random_file_index = np.random.randint(0,len(file_names))
            image_name = file_names[random_file_index]
            return cv2.imread(file_path+"/"+image_name)    
        
        for i in range(0,20):
            input_im = getRandomImage('test_set/')
            input_original = input_im.copy()
            input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            
            input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
            input_im = input_im / 255.
            input_im = input_im.reshape(1,224,224,3) 
            
            
            res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
            
            
            draw_test("Prediction", res, input_original) 
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
            
    def cluster_image_testing(self,name):
        
        
        classifier = load_model('Vggcd.h5') # loading the model
        input_im = cv2.imread(name)
        input_original = input_im.copy()
        input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
        
        input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
        input_im = input_im / 255.
        input_im = input_im.reshape(1,224,224,3) 
        
        
        res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
        
        if res==[0]:
            return True
            
        else:
            return False            
              
        cv2.waitKey(0)
                
    
    def processPictures(self):
        
        
        img_rows = 224
        img_cols = 224 
               
        vgg16 = VGG16(weights = 'imagenet', 
                         include_top = False, 
                         input_shape = (img_rows, img_cols, 3))
        
        
        
        for (i,layer) in enumerate(vgg16.layers):
            print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
            
        
        
        for layer in vgg16.layers:
            layer.trainable = False
            
        
        for (i,layer) in enumerate(vgg16.layers):
            print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
            
            
            
        def addTopModel(bottom_model, num_classes, D=256):
            top_model = bottom_model.output
            top_model = Flatten(name = "flatten")(top_model)
            top_model = Dense(D, activation = "relu")(top_model)
            top_model = Dropout(0.3)(top_model)
            top_model = Dense(num_classes, activation = "softmax")(top_model)
            return top_model
        
        
        
        
        num_classes = 2
        
        FC_Head = addTopModel(vgg16, num_classes)
        
        model = Model(inputs=vgg16.input, outputs=FC_Head)
        
        print(model.summary())
        
        
        
        
        
# Fitting the model to the images        
        train_data_dir = 'trainingset1'
        validation_data_dir = 'test_set'
        
        train_datagen = ImageDataGenerator(
              rescale=1./255,
              rotation_range=20,
              width_shift_range=0.2,
              height_shift_range=0.2,
              horizontal_flip=True,
              fill_mode='nearest')
         
        validation_datagen = ImageDataGenerator(rescale=1./255)
         
        
        train_batchsize = 16
        val_batchsize = 10
         
        train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size=(img_rows, img_cols),
                batch_size=train_batchsize,
                class_mode='categorical')
         
        validation_generator = validation_datagen.flow_from_directory(
                validation_data_dir,
                target_size=(img_rows, img_cols),
                batch_size=val_batchsize,
                class_mode='categorical',
                shuffle=False)
        
        
        
                           
        checkpoint = ModelCheckpoint("Vggcd.h5",
                                     monitor="val_loss",
                                     mode="min",
                                     save_best_only = True,
                                     verbose=1)
        
        earlystop = EarlyStopping(monitor = 'val_loss', 
                                  min_delta = 0, 
                                  patience = 3,
                                  verbose = 1,
                                  restore_best_weights = True)
        
        
        callbacks = [earlystop, checkpoint]
        
        
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = RMSprop(lr = 0.001),
                      metrics = ['accuracy'])
        
        nb_train_samples = 251
        nb_validation_samples = 273
        epochs = 3
        batch_size = 16
        
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = nb_train_samples // batch_size,
            epochs = epochs,
            callbacks = callbacks,
            validation_data = validation_generator,
            validation_steps = nb_validation_samples // batch_size)
        
        model.save("Vggcd.h5")
my_Model=PictureManager()
my_Model.processPictures()
my_Model.clusterImage()
