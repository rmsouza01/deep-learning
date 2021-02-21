import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf


seed = 909 # (IMPORTANT) to input image and corresponding target with same augmentation parameter.

gen_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',\
               "cval": 0}

train_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

train_target_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

train_image_generator = train_image_datagen.flow_from_directory("/home/roberto.medeirosdeso/Cluster-example/Data/Compressed/Train-root/",
                                                    class_mode=None, batch_size = 2,seed=seed, target_size=(288, 512),color_mode='rgb',shuffle = True)

train_target_generator = train_target_datagen.flow_from_directory("/home/roberto.medeirosdeso/Cluster-example/Data/Uncompressed/Train-root/",
                                                    class_mode=None, batch_size = 2, seed=seed, target_size=(288, 512),color_mode='rgb' ,shuffle = True)

train_generator = zip(train_image_generator, train_target_generator)

val_image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

val_target_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**gen_params) 

val_image_generator = val_image_datagen.flow_from_directory("/home/roberto.medeirosdeso/Cluster-example/Data/Compressed/Val-root/",
                                                     class_mode=None, batch_size = 2,seed=seed, target_size=(288, 512),color_mode='rgb',shuffle = True)

val_target_generator = val_target_datagen.flow_from_directory("/home/roberto.medeirosdeso/Cluster-example/Data/Uncompressed/Val-root/",
                                                    class_mode=None, batch_size = 2, seed=seed, target_size=(288, 512),color_mode='rgb' ,shuffle = True)


val_generator = zip(val_image_generator, val_target_generator)


seed = 909 # (IMPORTANT) to input image and corresponding target with same augmentation parameter.

gen_params = {"rescale":1.0/255,"featurewise_center":False,"samplewise_center":False,"featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,"zca_whitening":False,"rotation_range":20,"width_shift_range":0.1,"height_shift_range":0.1,\
              "shear_range":0.2, "zoom_range":0.1,"horizontal_flip":True,"fill_mode":'constant',\
               "cval": 0}


def get_unet_mod(patch_size = (288,512),learning_rate = 1e-3,\
                 learning_decay = 1e-6, drop_out = 0.1,nchannels = 3,kshape = (3,3)):
    ''' Get U-Net model with gaussian noise and dropout'''
    
    dropout = drop_out
    
    input_img = tf.keras.layers.Input((patch_size[0], patch_size[1],nchannels))
    
    conv1 = tf.keras.layers.Conv2D(48, kshape, activation='relu', padding='same')(input_img)
    conv1 = tf.keras.layers.Conv2D(48, kshape, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(96, kshape, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(96, kshape, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(192, kshape, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(192, kshape, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(384, kshape, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(384, kshape, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(dropout)(pool4)

    conv5 = tf.keras.layers.Conv2D(768, kshape, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(768, kshape, activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
    up6 = tf.keras.layers.Dropout(dropout)(up6)
    conv6 = tf.keras.layers.Conv2D(384, kshape, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(384, kshape, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3],axis=-1)
    up7 = tf.keras.layers.Dropout(dropout)(up7)
    conv7 = tf.keras.layers.Conv2D(192, kshape, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(192, kshape, activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2],axis=-1)
    up8 = tf.keras.layers.Dropout(dropout)(up8)
    conv8 = tf.keras.layers.Conv2D(96, kshape, activation='relu', padding='same')(up8)
    conv8 = tf.keras.layers.Conv2D(96, kshape, activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    up9 = tf.keras.layers.Dropout(dropout)(up9)
    conv9 = tf.keras.layers.Conv2D(48, kshape, activation='relu', padding='same')(up9)
    conv9 = tf.keras.layers.Conv2D(48, kshape, activation='relu', padding='same')(conv9)

    conv10 = tf.keras.layers.Conv2D(3, (1, 1), activation='linear')(conv9)
    out = tf.keras.layers.Add()([conv10, input_img]) 
    model = tf.keras.models.Model(inputs=input_img, outputs=out)
    opt = tf.keras.optimizers.Adam(lr= learning_rate, decay = learning_decay)
    model.compile(optimizer= opt,loss='mse')

    return model

model = get_unet_mod()
print(model.summary())

model_name = "/home/roberto.medeirosdeso/Cluster-example/unet_jpeg.h5"
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)

monitor = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',\
                                             verbose=0,save_best_only=True,\
                                             save_weights_only=True,\
                                             mode='min')
# Learning rate schedule
def scheduler(epoch, lr):
    if epoch%3 == 0 and epoch!= 0:
        lr = lr/2
    return lr

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)

history = model.fit(train_generator, steps_per_epoch=1000, validation_data = (val_generator),\
                    validation_steps = 500,\
                    epochs=15,verbose=1, callbacks = [early_stop, monitor, lr_schedule])

import natsort
model = get_unet_mod(patch_size = (None,None))
model.load_weights(model_name)
test_compressed = np.array(glob.glob("/home/roberto.medeirosdeso/Cluster-example/Data/Compressed/Test-root/Test/*.jpg"))
test_uncompressed = np.array(glob.glob("/home/roberto.medeirosdeso/Cluster-example/Data/Uncompressed/Test-root/Test/*.png"))

test_compressed = natsort.natsorted(test_compressed)
test_uncompressed = natsort.natsorted(test_uncompressed)

X_test = np.zeros((100,256,256,3))
Y_test = np.zeros((100,256,256,3))

for ii in range(100):
    X_test[ii] = np.array(tf.keras.preprocessing.image.load_img(test_compressed[ii]))[:256,:256,:]/255.0
    Y_test[ii] = np.array(tf.keras.preprocessing.image.load_img(test_uncompressed[ii]))[:256,:256,:]/255.0
    
Ypred = np.clip(model.predict(X_test,batch_size = 2),0,1)

MSE1 = ((Ypred - Y_test)**2).mean()
MSE2 = ((X_test - Y_test)**2).mean()
print("UNET MSE:")
print(MSE1)
print("JPEG MSE:")
print(MSE2)
