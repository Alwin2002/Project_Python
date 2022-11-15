from cnn import unet
from main import batch_generator
from keras.callbacks import ModelCheckpoint
import random
import os

image_list=os.listdir('Datasets/PageSegData/PageImg/')
image_list=[filename.split(".")[0]for filename in image_list]

random.shuffle(image_list)
file_train=image_list[0:int(0.75*len(image_list))]
file_test=image_list[int(0.75*len(image_list)):]


def training():
    model = unet()
    mc = ModelCheckpoint('weights{epoch:08d}.h5', save_weights_only=True, period=1)
    model.fit(batch_generator(file_train,2,2),epochs=3,steps_per_epoch=1000,
                    validation_data=batch_generator(file_test,2,2),
                    validation_steps=400,callbacks=[mc],shuffle=1)