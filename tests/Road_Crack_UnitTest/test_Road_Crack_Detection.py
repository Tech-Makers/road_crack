
from keras.models import Sequential
sys.path.append('../src/')
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.applications import VGG16
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


from Road_Crack_Detection import  PictureManager 

import unittest

class test_Picture_Manager(unittest.TestCase):
    def test_cluster_image_testingcrack(self):
        c = PictureManager()
        
        self.assertEqual(c.cluster_image_testing("crack.jpg"),
                        True)
        
    def test_cluster_image_testinguncrack(self):
        c = PictureManager()
        
        self.assertEqual(c.cluster_image_testing("non_crack.jpg"),
                        False)
        
        
    
if __name__ == '__main__':
	unittest.main() 
