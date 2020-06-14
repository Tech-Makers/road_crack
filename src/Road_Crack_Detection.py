# Importing the libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import boto3
import os 
import sys 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import shutil
from os import walk
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from os import walk
import zipfile
from datetime import datetime

class Delete_Manager:
    def thedelete(self) :
        
        dir_path1 = 'F:\crack\crack_clustered'
        
        try:
            shutil.rmtree(dir_path1)
        except OSError as e:
            print("Error: %s : %s" % (dir_path1, e.strerror))
        print("crack clustered deleted")

       
        
        
        
        Y = []
        
        for (dirpath, dirnames, filenames) in walk("F:\\crack\\my_dataset_final"):
            Y.extend(filenames)
            break
        
        for i in range(0,len(Y)):
            os.remove("F:\\crack\\my_dataset_final\\" + Y[i])
       

        dir_path2 = 'F:\crack\my_dataset_final'
        
        try:
            shutil.rmtree(dir_path2)
        except OSError as e:
            print("Error: %s : %s" % (dir_path2, e.strerror))
        print("dataset deleted")
         
        
        dir_path3 = 'F:\\crack\\non_crack_clustered'
        
        try:
            shutil.rmtree(dir_path3)
        except OSError as e:
            print("Error: %s : %s" % (dir_path3, e.strerror))
        print("non crack deleted")
        
        
        
        
        dir_path4 = 'F:\\crack\\RAR_File'
        
        try:
            shutil.rmtree(dir_path4)
        except OSError as e:
            print("Error: %s : %s" % (dir_path4, e.strerror))
        print("non crack deleted")
           
            
            
class Upload_Manager:
    def uploader(self,filename):
        with open(filename , "rb") as f:
           data = f.read()
          
        client = boto3.client('s3')
        response = client.put_object(
            ACL='private',
            Body=data,
            Bucket='souvikhelloworld',
            Key=filename
        )   
        print("uploaded sucessfully on to the server")
    
    
        
            
 

               
#This function has to be visited
class Input_Manager:
    def __init__(self):
        
        directory1 = "my_dataset_final"
        directory2 = "crack_clustered" 
        directory3="non_crack_clustered"
        directory4="RAR_File"
        
        parent_dir = "F:\\crack\\"
          
         
        path1 = os.path.join(parent_dir, directory1) 
        path2 = os.path.join(parent_dir, directory2) 
        path3 = os.path.join(parent_dir, directory3)
        path4 = os.path.join(parent_dir, directory4) 
        os.mkdir(path1) 
        os.mkdir(path2) 
        os.mkdir(path3)
        os.mkdir(path4)
        
        print("Directory1 '% s' created" % directory1) 
        print("Directory2 '% s' created" % directory2) 
        print("Directory3 '% s' created" % directory3) 
        print("Directory3 '% s' created" % directory4) 
        
        client = boto3.client('s3')
        client.download_file('souvikhelloworld','files.zip','F:\\crack\\my_dataset_final\\images.zip')
        with zipfile.ZipFile('F:\\crack\\my_dataset_final\\images.zip','r') as my_zip:
            my_zip.extractall('F:\\crack\\my_dataset_final\\images')
        
        


class bucket_creater:
    def __init__(self):
        client = boto3.client('s3')
        response = client.create_bucket(ACL='private',
                                        Bucket='souvikhelloworld',
                                        CreateBucketConfiguration={
                                            'LocationConstraint': 'ap-south-1'
                                        }
                                       )


class Report_Manager:
    def mail(self,to, subject, text, attach):
        if not isinstance(to,list):
            to = [to]
        
        if not isinstance(attach,list):
            attach = [attach]
        gmail_user='#######@gmail.com'
        gmail_pwd = "########"
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = ", ".join(to)
        msg['Subject'] = 'CRACK REPORT GENERATED'
    
        msg.attach(MIMEText(text))
    
        
        for file in attach:
            
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(file, 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(file))
            msg.attach(part)
    
        mailServer = smtplib.SMTP("smtp.gmail.com", 587)
        mailServer.ehlo()
        mailServer.starttls()
        mailServer.ehlo()
        mailServer.login(gmail_user, gmail_pwd)
        mailServer.sendmail(gmail_user, to, msg.as_string())
        mailServer.close()
        print("mail generated")
    
    def report(self):
        
        G = []
        
        for (dirpath, dirnames, filenames) in walk("F:\\crack\\crack_clustered"):
            G.extend(filenames)
            break
        
        T=len(G)
        
        for i in range(0,T):
            G[i]="F:\\crack\\crack_clustered\\" + G[i]
        
        now = datetime.now()
        dt_string = now.strftime("%d.%m.%Y %H.%M.%S")
              
        with zipfile.ZipFile("F:\\crack\\RAR_File\\"+ dt_string+".zip","w",compression=zipfile.ZIP_DEFLATED) as my_zip:
            for i in G:
                my_zip.write(i)
        print("zipping done")
        
        print(G)
        
        
        M = []
        
        for (dirpath, dirnames, filenames) in walk("F:\\crack\\RAR_File"):
            M.extend(filenames)
            break
        
        T=len(M)
        
        for i in range(0,T):
            M[i]="F:\\crack\\RAR_File\\" + M[i]
            print(M)
        
        subject='Road Crack data Report'
        bodytext='This is the report we get after analysing the data an urgent action is to be taken on the following data'

        
        self.mail(['#####@gmail.com','######@gmail.com'], subject, bodytext,M)
        print("Report Created")
        
                


class Picture_Manager(Input_Manager,Upload_Manager):
    
    def place_file_to_folder(self,file_name, folder_name):
        os.makedirs(folder_name, mode = 0o777, exist_ok = True)
        shutil.copy2(file_name, folder_name)
        
    def image_status_crack(self,filename):
        
        self.place_file_to_folder(filename, "F:\\crack\\crack_clustered")
        print("Image moved successfully to crack_clustered ")
        
    
    def image_status_noncrack(self,filename):
        self.place_file_to_folder(filename, "F:\\crack\\non_crack_clustered")
        print("Image moved successfully to non_crack_clustered ")
        
    def cluster_image_Version_one(self,name):
        classifier = load_model('Vggcd.h5')
        input_im = cv2.imread(name)
        input_original = input_im.copy()
        input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
        
        input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
        input_im = input_im / 255.
        input_im = input_im.reshape(1,224,224,3) 
        
        
        res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
        
        if res==[0]:
            return True
            print ("True")
            self.uploader(name)
            self.image_status_crack(name)
            
            
            
        else:
            return False
            print ("False")
            self.image_status_noncrack(name)
            
        
        
        cv2.waitKey(0)
        
    
   


    
    
    def cluster_image_version_zero(self):
        classifier = load_model('Vggcd.h5')
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
        
        for i in range(0,50):
            input_im = getRandomImage("testset_final/")
            input_original = input_im.copy()
            input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
            
            input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
            input_im = input_im / 255.
            input_im = input_im.reshape(1,224,224,3) 
            
            
            res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
            
            
            draw_test("Prediction", res, input_original) 
            cv2.waitKey(0)
            
            cv2.destroyAllWindows()
    
    def Process_picture(self):
        
        
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
        
        
        
        
        
        
        train_data_dir = 'trainingset1'
        validation_data_dir = 'testset1'
        
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
        
        
        
                           
        checkpoint = ModelCheckpoint("Vggcdlaptop.h5",
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
        nb_validation_samples = 160
        epochs = 3
        batch_size = 16
        
        history = model.fit_generator(
            train_generator,
            steps_per_epoch = nb_train_samples // batch_size,
            epochs = epochs,
            callbacks = callbacks,
            validation_data = validation_generator,
            validation_steps = nb_validation_samples // batch_size)
        
        model.save("Vggcdlaptop.h5")
        


my_Model=Picture_Manager()
#my_Model.Process_picture()
#my_Model.cluster_image_Version_one()


f = []
for (dirpath, dirnames, filenames) in walk("F:\crack\my_dataset_final\images\crack\my_dataset_final"):
    f.extend(filenames)
    break

for i in range(0,len(f)):
    f[i]="F:\\crack\\my_dataset_final\\images\\crack\my_dataset_final\\" + f[i]

for i in f:
    my_Model.cluster_image_Version_one(i)
print("Both the folder created now preparing to send the mail")
r=Report_Manager()
r.report()

d=Delete_Manager()
d.thedelete()
print("deleted everything")
