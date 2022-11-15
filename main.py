import numpy as np
import matplotlib.pyplot as plt
import cv2
from train_model import training
import random



def visualize(img,seg_img):
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(1,2,2)
    plt.imshow(seg_img)
    plt.title('Segmented Image')
    plt.show()


def preprocess_img(img):
    img=cv2.resize(img,(512,512))
    return img


def batch_generator(filelist,n_classes,batch_size):
     while True:
        X=[]
        Y=[]
        for i in range(batch_size):

            fn=random.choice(filelist)

            img=cv2.imread(f'Datasets/PageSegData/PageImg/{fn}.JPG',0)

            ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)

            img=cv2.resize(img,(512,512))

            img=np.expand_dims(img,axis=-1)

            img=img/255

            seg=cv2.imread(f'Datasets/PageSegData/PageSeg/{fn}_mask.png',1)
            seg=get_segmented_img(seg,n_classes)

            X.append(img)
            Y.append(seg)
     yield np.array(X),np.array(Y)


def get_segmented_img(img, n_classes):
    seg_labels = np.zeros((512, 512, 1))
    img = cv2.resize(img, (512, 512))
    img = img[:, :, 0]
    cl_list = [0, 24]
    seg_labels[:, :, 0] = (img != 0).astype(int)
    return seg_labels

training()





