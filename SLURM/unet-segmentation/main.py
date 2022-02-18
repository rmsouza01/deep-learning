from utils import prepare_data
from models import unet, dice_coef_loss, dice_coef
import glob
from data_generator import DataGeneratorUnet
import tensorflow as tf


def run_data_preparation():
    # Training set preparation
    files_list_train = "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\train-set.txt"
    imgs_paths = ["C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\Original",
                  "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\Skull-stripping-masks",
                  "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\WM-GM-CSF"]
    out_paths_train = ["C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Train\\Image",
                       "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Train\\Brain-mask",
                       "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Train\\WM-GM-CSF"
                       "-mask"]

    prepare_data(files_list_train, imgs_paths, out_paths_train, verbose=1)

    # Validation set preparation
    files_list_val = "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\val-set.txt"
    out_paths_val = ["C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Validation\\Image",
                     "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Validation\\Brain"
                     "-mask",
                     "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Validation\\WM-GM"
                     "-CSF-mask"]

    prepare_data(files_list_val, imgs_paths, out_paths_val, verbose=1)
    return


def run_training():
    imgs_list_train = glob.glob(
        "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Train\\Image\\*.npy")
    masks_list_train01 = glob.glob(
        "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\pre-processed\\Train\\Brain-mask\\*.npy")
    masks_list_train02 = glob.glob(
        "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\pre-processed\\Train\\WM-GM-CSF-mask\\*.npy")

    imgs_list_val = glob.glob(
        "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\\pre-processed\\Validation\\Image\\*.npy")
    masks_list_val01 = glob.glob(
        "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\pre-processed\\Validation\\Brain-mask\\*.npy")
    masks_list_val02 = glob.glob(
        "C:\\Users\\rober\\PycharmProjects\\unet-segmentation\\Data\pre-processed\\Validation\\WM-GM-CSF-mask\\*.npy")

    batch_size = 64
    gen_train = DataGeneratorUnet(imgs_list_train, masks_list_train01, masks_list_train02, batch_size=batch_size)
    gen_val = DataGeneratorUnet(imgs_list_val, masks_list_val01, masks_list_val02, batch_size=batch_size)

    model = unet(input_shape=(128, 128, 1))
    print(model.summary())

    # Callbacks
    model_name = "unet_multitask.h5"

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    monitor = tf.keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss',
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='min')

    def scheduler(epoch, lr):
        if epoch % 5 == 0 and epoch != 0:
            lr = lr / 2
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

    # Compile and train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=[dice_coef_loss, dice_coef_loss], loss_weights=[0.5, 0.5], metrics=[dice_coef])

    model.fit(gen_train, epochs=100, verbose=1,
              callbacks=[early_stop, monitor, lr_schedule],
              validation_data=(gen_val),
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,)

    return


if __name__ == '__main__':
    prepare_flag = False
    train_flag = True

    if prepare_flag:
        run_data_preparation()

    if train_flag:
        run_training()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
