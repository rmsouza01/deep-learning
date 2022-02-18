import numpy as np
import os
import nibabel as nib

def bbox3D(img):
    #include your solution below
    y,x,z = np.nonzero(img)
    r1, c1, d1 = min(x), min(y), min(z)
    r2, c2, d2 = max(x), max(y), max(z)
    return (r1,c1,d1),(r2,c2,d2)


def remove_blank_slices(imgs):
    imgs_cropped = []

    bin_img = imgs[0] > 0.1*imgs[0].min()
    (r1, c1, d1), (r2, c2, d2) = bbox3D(bin_img)
    for ii in range(len(imgs)):
        imgs_cropped.append(imgs[ii][r1:r2, c1:c2,d1:d2])
    return imgs_cropped

def min_max_normalization(data):
    data_min, data_max = data.min(), data.max()
    data_norm = (data - data_min)/(data_max - data_min)
    return data_norm


def prepare_data(files_list_txt, imgs_paths, out_paths, verbose = 1):
    counter = 1
    train_set = np.loadtxt(files_list_txt,dtype = str)
    for ii in train_set:
        ss_mask = ii.split(".nii")[0] + "_staple.nii.gz"

        file_name = os.path.join(imgs_paths[0], ii + ".gz")
        mask_name01 = os.path.join(imgs_paths[1],ss_mask)
        mask_name02 =  os.path.join(imgs_paths[2], ii + ".gz")


        data = nib.load(file_name).get_fdata()
        mask01 = nib.load(mask_name01).get_fdata()
        mask02 = nib.load(mask_name02).get_fdata()

        data_norm = min_max_normalization(data)
        cropped_imgs = remove_blank_slices([data_norm, mask01, mask02])
        data_cropped = cropped_imgs[0]
        mask01_cropped = cropped_imgs[1]
        mask02_cropped = cropped_imgs[2]
        if verbose:
            print(data_norm.shape)
            print(data_cropped.shape)

        H = data_cropped.shape[2]
        for jj in range(H):
            data_slice = data_cropped[:,:,jj]
            mask01_slice = mask01_cropped[:,:,jj]
            mask02_slice = mask02_cropped[:,:,jj]

            np.save(os.path.join(out_paths[0], "img_" + str(counter) + ".npy"), data_slice)
            np.save(os.path.join(out_paths[1], "img_" + str(counter) + ".npy"), mask01_slice)
            np.save(os.path.join(out_paths[2], "img_" + str(counter) + ".npy"), mask02_slice)

            counter+=1
    return