import fnmatch
import cv2
import numpy as np
import string
import time
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from PIL import Image





char_list = string.ascii_letters+string.digits

def encode_to_labels(txt):
   
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

def find_dominant_color(image):

        width, height = 150,150
        image = image.resize((width, height),resample = 0)
 
        pixels = image.getcolors(width * height)

        sorted_pixels = sorted(pixels, key=lambda t: t[0])

        dominant_color = sorted_pixels[-1][1]
        return dominant_color

def preprocess_img(img, imgSize):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]]) 
        print("Image None!")

   
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))  
    img = cv2.resize(img, newSize, interpolation=cv2.INTER_CUBIC) 
                                                              
    most_freq_pixel=find_dominant_color(Image.fromarray(img))
    target = np.ones([ht, wt]) * most_freq_pixel  
    target[0:newSize[1], 0:newSize[0]] = img

    img = target

    return img



training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []
max_label_len = 0

annot=open('/content/Data-generator-for-CRNN/annotation.txt','r').readlines()
imagenames=[]
txts=[]

for cnt in annot:
    filename,txt=cnt.split(',')[0],cnt.split(',')[1].split('\n')[0]
    imagenames.append(filename)
    txts.append(txt)
    
c = list(zip(imagenames, txts))

random.shuffle(c)

imagenames, txts = zip(*c)
    

    
for i in range(len(imagenames)):
        img = cv2.imread('/content/Data-generator-for-CRNN/images/'+imagenames[i],0)   
 
        img=preprocess_img(img,(128,32))
        img=np.expand_dims(img,axis=-1)
        img = img/255.
        txt = txts[i]
        
        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)
            
           

        if i%10 == 0:     
            valid_orig_txt.append(txt)   
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)   
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt)) 
        
        # break the loop if total data is 150000
        if i == 150000:
            flag = 1
            break
        i+=1