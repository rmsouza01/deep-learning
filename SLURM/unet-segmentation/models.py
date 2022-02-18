import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


# TensorFLow
def dice_coef(y_true, y_pred):
    ''' Metric used for CNN training'''
    smooth = 1.0 #CNN dice coefficient smooth
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    ''' Loss function'''
    return -(dice_coef(y_true, y_pred)) # try negative log of the DICE* (-tf.math.log())



def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor

    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters,
                                   kernel_size=kernel_size, padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    return x


def enconder_block(inputs, n_filters, pool_size, dropout):
    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size)(f)
    p = tf.keras.layers.Dropout(dropout)(p)

    return f, p


def encoder(inputs):
    f1, p1 = enconder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.0)
    f2, p2 = enconder_block(p1, n_filters=128, pool_size=(2, 2), dropout=0.2)
    f3, p3 = enconder_block(p2, n_filters=256, pool_size=(2, 2), dropout=0.2)
    f4, p4 = enconder_block(p3, n_filters=512, pool_size=(2, 2), dropout=0.2)

    return p4, (f1, f2, f3, f4)


def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)

    return bottle_neck


def decoder_block(inputs, conv_output, n_filters, kernel_size, dropout):
    u = tf.keras.layers.UpSampling2D()(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)

    return c


def decoder(inputs, convs):
    f1, f2, f3, f4 = convs

    d1 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3, 3), dropout=0.0)
    d2 = decoder_block(d1, f3, n_filters=256, kernel_size=(3, 3), dropout=0.2)
    d3 = decoder_block(d2, f2, n_filters=128, kernel_size=(3, 3), dropout=0.2)
    d4 = decoder_block(d3, f1, n_filters=64, kernel_size=(3, 3), dropout=0.2)

    output01 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
    output02 = tf.keras.layers.Conv2D(4, (1, 1), activation='softmax')(d4)

    return output01, output02


def unet(input_shape=(None, None, 1)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    output01, output02 = decoder(bottle_neck, convs)

    model = tf.keras.Model(inputs=inputs, outputs=[output01, output02])

    return model
