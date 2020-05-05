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

class PictureManager(object):

  def processPictures(self):

    img_width = 150
    img_height = 150

    train_data_dir = '/content/drive/My Drive/Road/training_set'
    validation_data_dir = '/content/drive/My Drive/Road/validation_set'
    train_samples = 8000
    validation_samples = 2000
    epochs = 2
    batch_size = 20


    if K.image_data_format() == 'channels_first':
      input_shape = (3, img_width, img_height)
    else:
      input_shape = (img_width, img_height, 3)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',                                                                                  
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))    

    model.compile(loss='binary_crossentropy',                
                  optimizer='adam',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(                
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(        
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    print(train_generator.class_indices)

    imgs, labels = next(train_generator)

    from skimage import io

    def imshow(image_RGB):
      io.imshow(image_RGB)
      io.show()

    import matplotlib.pyplot as plt
    %matplotlib inline
    image_batch,label_batch = train_generator.next()

    print(len(image_batch))
    for i in range(0,len(image_batch)):
      image = image_batch[i]
      print(label_batch[i])
      imshow(image)

    validation_generator = test_datagen.flow_from_directory(                        
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    history = model.fit_generator(        
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size)

    model.save('/content/drive/My Drive/Road/save.h5')
    print("saved")

    import matplotlib.pyplot as plt
    %matplotlib inline

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
  
  def test_clusterImage(self,name):
    classifier = load_model('/content/drive/My Drive/Road/save.h5')
   
    # predicting images
    from keras.preprocessing import image
    noncrack_counter = 0 
    crack_counter  = 0
    img_width = 150
    img_height = 150

    img = image.load_img(name, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
        
    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=20)
    classes = classes[0][0]
        
    if classes == 0:
      return True
      print('Cracks')
    else:
      return False
      print('Non_cracks')

  def clusterImage(self):
    classifier = load_model('/content/drive/My Drive/Road/save.h5')
    ## Now Predict
    predict_dir_path='/content/drive/My Drive/Road/test_set/'
    onlyfiles = [f for f in listdir(predict_dir_path) if isfile(join(predict_dir_path, f))]
    print(onlyfiles)

    # predicting images
    from keras.preprocessing import image
    noncrack_counter = 0 
    crack_counter  = 0
    img_width = 150
    img_height = 150
    for file in onlyfiles:
        img = image.load_img(predict_dir_path+file, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        images = np.vstack([x])
        classes = classifier.predict_classes(images, batch_size=20)
        classes = classes[0][0]
        
        if classes == 0:
            print(file + ": " + 'Cracks')
            crack_counter += 1
        else:
            print(file + ": " + 'Non_cracks')
            noncrack_counter += 1
    print("Total NonCracks :",noncrack_counter)
    print("Total Cracks :",crack_counter)

my_Model=PictureManager()
my_Model.processPictures()
my_Model.clusterImage()
my_Model.test_clusterImage('/content/drive/My Drive/Road/test_set/00002.jpg')
