
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import keras
from tensorflow.python.keras import optimizers


from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join

from Road_Crack_Detection import  PictureManager 

import unittest

class test_Picture_Manager(unittest.TestCase):
    def test_clusterImage_testingcrack(self):
        c = PictureManager()
        
        self.assertEqual(c.test_clusterImage('/content/drive/My Drive/Road/test_set/00003.jpg'),
                        True)
    def test_clusterImage_testing(self):
        c = PictureManager()
        
        self.assertEqual(c.test_clusterImage('/content/drive/My Drive/Road/test_set/00002.jpg'),
                        True)   
              
    
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
