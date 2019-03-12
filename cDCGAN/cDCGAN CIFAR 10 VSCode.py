#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'cDCGAN'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # cDCGAN (conditional Deep Convolutional GAN)
# 
# **Author:** Teebone Ding
# 
# **Date:** 2019 Feb
# 
# Based on DCGAN (Radford, Metz, 2016) [Paper Link](https://arxiv.org/pdf/1511.06434.pdf)
# 
# Some useful refs while I implementing my cDCGAN:
# * keras GAN [Link](https://github.com/eriklindernoren/Keras-GAN/blob/master/cgan/cgan.py)
# * cDCGAN in tensorflow version [Link](https://github.com/znxlwm/tensorflow-MNIST-cGAN-cDCGAN)
# * cDCGAN using keras [Link](https://medium.com/@utk.is.here/training-a-conditional-dc-gan-on-cifar-10-fce88395d610)
# 
# Using **CIFAR 10 dataset** as playground.
# 
# My SW/HW setting:
# * ASUS nVidia GTX 1060 3GB
# * RAM: 16GB
# * CPU: Intel Core i5-7400 3.00GHz
# * Ubuntu 16.04 LTS

#%%
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
# Check GPU device
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


#%%
print(tf.__version__)

#%% [markdown]
# ## Load CIFAR 10 data and data explore

#%%
# Load CIFAR 10 data
cifar10_dataset = keras.datasets.cifar10
(x_train, y_train), (_, _) = cifar10_dataset.load_data()


#%%
print(x_train.shape)
print(y_train.shape)


#%%
# Ref: https://github.com/EN10/CIFAR
y_cat = {
    0 : "airplane",
    1 : "automobile",
    2 : "bird",
    3 : "cat",
    4 : "deer",
    5 : "dog",
    6 : "frog",
    7 : "horse",
    8 : "ship",
    9 : "truck"
}


#%%
from tensorflow.keras.utils import to_categorical
y_train_one_hot = to_categorical(y_train)


#%%
print(y_train_one_hot.shape)
print(y_train_one_hot[:10])


#%%
y_train[:10]


#%%
x_train[0]

#%% [markdown]
# # Preprocess training data

#%%
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
#x_train = x_train.astype(np.float32)/255.0


#%%
x_train.shape

#%% [markdown]
# ## Build Generator and Discriminator
# Try CNN layers to build GAN (cDCGAN).

#%%
from tensorflow.keras.layers import Input, Dense, Concatenate, Reshape, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, ReLU, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


#%%
z = Input(shape=(100,))    # Normal distribution noise input
x = Input(shape=(32,32,3)) # image, by real dataset or generator
y = Input(shape=(10,))     # label, MNIST one hot encoded with 10 labels


#%%
def build_generator(z,y):
    layer = Concatenate()([z,y])
    
    layer = Dense((4*4*256))(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = Reshape((4,4,256))(layer)
    
    layer = Conv2DTranspose(128,5,strides=2,padding='same')(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2DTranspose(64,5,strides=2, activation='relu',padding='same')(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2DTranspose(3,5,strides=2, activation='tanh',padding='same')(layer)
    
    model = Model([z,y],layer)

    return model,layer

G,G_out = build_generator(z,y)
G.summary() # not compiled, combined with discriminator and compile.


#%%
def build_discriminator(x,y):
    layer = Conv2D(32,3,strides=2,padding="same")(x)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = Dropout(0.25)(layer)
    
    layer = Conv2D(64,3,strides=2,padding="same")(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = Dropout(0.25)(layer)
    
    layer = Conv2D(128,3,strides=2,padding="same")(layer)
    layer = BatchNormalization(momentum=0.8)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)
    layer = Dropout(0.25)(layer)
    
    layer = Flatten()(layer)
    layer = Concatenate()([layer, y])
    layer = Dense(256,activation='relu')(layer)
    
    layer = Dense(1,activation='sigmoid')(layer)
    
    model = Model([x,y],layer)
    return model, layer

D, D_out = build_discriminator(x,y)
#D.summary()


#%%
# While training....
# 1. train discriminator first
# 2. train a combined (G+D) model, with D is not trainable (only train G)
# 3. calculate loss value and save weights


#%%
# Compile Discriminator model
#sgd = SGD(lr=0.1, momentum=0.5 ,decay= 1.00004)
adam = Adam(0.0002, 0.5)
D.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
D.summary()


#%%
# Compile combined G+D model with D is not trainable
img = G([z,y])
D.trainable = False
valid = D([img,y])
C = Model([z,y], valid)
C.summary()
C.compile(optimizer=adam,loss='binary_crossentropy')

#%% [markdown]
# # Train conditional GAN

#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)


#%%
# dataset: x_train, y_train_one_hot

#%%
!mkdir images
!mkdir models

#%%
# Randomly generate images from Generator network
def sample_gen_imgs(epoch):
    gen_labels = np.repeat(np.arange(10),10)
    gen_labels_cat = to_categorical(gen_labels, num_classes=10)
    noises = np.random.normal(0,1,(100,100))
    gen_imgs = G.predict([noises,gen_labels_cat])
    gen_imgs = 0.5 * gen_imgs + 0.5

    r,c = 10,10
    fig, axs = plt.subplots(r, c, figsize=(20, 20))
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,:])
            axs[i,j].set_title("Cat: %s" % y_cat[gen_labels[cnt]])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/gen_cifar10_%d.png"%epoch)
    plt.close()


#%%
import pdb
import datetime
def train(epochs = 10 ,batch_size = 128): 
    print("[%s] Start training..."%(datetime.datetime.now()))
    real_label = np.ones((batch_size,1))
    fake_label = np.zeros((batch_size,1))
    
    for epoch in range(epochs):
        epoch += 1
        for _ in range(int(x_train.shape[0]/batch_size)):
            # Train Discriminator once
            # sample real data from training dataset
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs, labels = x_train[idx], y_train_one_hot[idx]
            # sample noises from normal dist.
            # Somehow I tried uniform dist but cannot trained very well.
            noises = np.random.normal(0,1,(batch_size,100))
            # generate fake images
            gen_imgs = G.predict([noises,labels])
            #pdb.set_trace()
            # Train Discriminator
            d_real_loss = D.train_on_batch([imgs,labels], real_label) 
            d_fake_loss = D.train_on_batch([gen_imgs,labels],fake_label)
            d_loss = 0.5*np.add(d_real_loss,d_fake_loss)
           
            # Train Combined model (generator)
            gen_labels = to_categorical(np.random.randint(0,10, batch_size),num_classes=10)
            # train generator
            g_loss = C.train_on_batch([noises,gen_labels], real_label)
            print("[%s] D-Loss value: %.5f Acc: %.5f G-Loss value: %.5f in epoch: %d"%(datetime.datetime.now(),d_loss[0],d_loss[1], g_loss, epoch),end='\r')
        
        if epoch % 5 == 0:
            print("[%s] D-Loss value: %.5f Acc: %.5f G-Loss value: %.5f in epoch: %d"%(datetime.datetime.now(),d_loss[0],d_loss[1], g_loss, epoch))
            sample_gen_imgs(epoch)
            G.save("models/G_cifar10_%d.h5"%epoch)
            C.save("models/C_cifar10_%d.h5"%epoch)
            D.save("models/D_cifar10_%d.h5"%epoch)


#%%
train(epochs=10, batch_size = 128)

#%% [markdown]
# ## Generated images after 10 epochs
# ![alt text](images/gen_cifar10_10.png "Gen image 10 epochs")
 