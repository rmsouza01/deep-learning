import random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def one_hot_seg(mask, nclasses):
    mask_oh = np.zeros((mask.shape[0],mask.shape[1],nclasses), dtype = int)
    for ii in range(1, nclasses):
        mask_oh[mask == ii, ii] = ii
    return mask_oh


class DataGeneratorUnet(tf.keras.utils.Sequence):
    'Generates data for Keras'
    'In order show clearly our applied augmentations we change default patch size to near real image size but we can ' \
    'change it to 128*128'

    def __init__(self, imgs_list, masks_list01, masks_list02, patch_size=(128, 128), batch_size=48, shuffle=True):

        self.imgs_list = imgs_list
        self.masks_list01 = masks_list01
        self.masks_list02 = masks_list02

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.number_of_samples = len(imgs_list)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.imgs_list) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, Y = self.__data_generation(batch_indexes)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.number_of_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'

        # Initialization
        X = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1))
        Y1 = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 1)) # Brain masks
        Y2 = np.empty((self.batch_size, self.patch_size[0], self.patch_size[1], 4)) # WM, GM, CSF masks

        for (jj, ii) in enumerate(batch_indexes):
            aux_img = np.load(self.imgs_list[ii])
            aux_mask01 = np.load(self.masks_list01[ii])
            aux_mask02 = np.load(self.masks_list02[ii])

            # Implement data augmentation function
            #img_aug, mask_aug01, mask_aug02 = self.__data_augmentation(aux_img, aux_mask01, aux_mask02)

            aux_img_patch, aux_mask_patch01, aux_mask_patch02 = self.__extract_patch(aux_img, aux_mask01, aux_mask02)

            X[jj,:,:,0] = aux_img_patch
            Y1[jj,:,:,0] = aux_mask_patch01
            aux_mask_patch02_oh = one_hot_seg(aux_mask_patch02,4)
            Y2[jj] = aux_mask_patch02_oh
        return X, [Y1, Y2]


    # Extract patch from image and mask
    def __extract_patch(self, img, mask01, mask02):
        crop_idx = [None] * 2
        #print(img.shape)
        crop_idx[0] = np.random.randint(0, img.shape[0] - self.patch_size[0])
        crop_idx[1] = np.random.randint(0, img.shape[1] - self.patch_size[1])
        img_cropped = img[crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                      crop_idx[1]:crop_idx[1] + self.patch_size[1]]
        mask_cropped01 = mask01[crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                         crop_idx[1]:crop_idx[1] + self.patch_size[1]]

        mask_cropped02 = mask02[crop_idx[0]:crop_idx[0] + self.patch_size[0], \
                         crop_idx[1]:crop_idx[1] + self.patch_size[1]]

        return img_cropped, mask_cropped01, mask_cropped02
