from os import listdir, remove
from os.path import isfile, join
from tabnanny import verbose

import numpy as np

import matplotlib.pyplot as plt
import cv2

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_aurelie_test():
    aurelie_test = [r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (1).jpeg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (1).JPG', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (23).jpeg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (75).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (113).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (115).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (473).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (702).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (719).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (808).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (825).JPG', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\dog.001 (902).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\dog\cat_dog.001 (721).jpg',
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\cat\cat.001 (1).jpeg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\cat\cat.001 (16).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\cat\cat.001 (18).jpg', 
    r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set\cat\cat.001 (32).jpg']

    aurelie_y = [1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,0, 0, 0, 0]
    return aurelie_test, aurelie_y 

def get_aurelie_full_test(label_codes, verbose=0):

    aurelie_test = get_dir_files(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set', include_sub_dir=1, verbose=verbose)
    aurelie_y = []

    for f in aurelie_test:
        category=f.split('\\')[-1]
        category=category.split('.')[0]
        aurelie_y.append(label_codes.get(category,0))
        
    return aurelie_test, aurelie_y 


def get_cv2_data(data_path, labels, image_size = (180, 180), verbose=0):
    data = [] 
    filenames = get_dir_files(dir_path=data_path, include_sub_dir=1, verbose=verbose-1)

    for f_name in filenames:

        try:
            img_arr = cv2.imread(join(data_path, f_name))[...,::-1] #convert BGR to RGB format
            resized_arr = cv2.resize(img_arr, image_size) # Reshaping images to preferred size
            
            # Affectation de la catégorie en se basant sur le nom du fichier
            cat = 0
            category=f_name.split('\\')[-1]
            category=category.split('.')[0]
            # On affecte le bon code de label
            for i in range(1, len(labels)):
                if labels[i] in category:
                    cat = i
                    break
            data.append([resized_arr, cat])
        except Exception as e:
            print(e)
    return np.array(data, dtype=object)


def get_dataset(path, image_size=(256, 256), batch_size = 32, color_mode='grayscale'):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels='inferred',
        label_mode='int',
        class_names=None,
        color_mode=color_mode, # "grayscale", "rgb", "rgba"
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
    )


def make_model(input_shape, num_classes, data_augmentation=None, nb_dim=3):
    inputs = keras.Input(shape=input_shape)

    x = inputs

    # Image augmentation block
    # With this option, your data augmentation will happen on device, synchronously with the rest of the model execution, meaning that it will benefit from GPU acceleration.
    # Note that data augmentation is inactive at test time, so the input samples will only be augmented during fit(), not when calling evaluate() or predict().
    # If you're training on GPU, this is the better option.
    if data_augmentation:
        x = data_augmentation(x)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, nb_dim, strides=2, padding="same", input_shape=input_shape)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, nb_dim, padding="same", input_shape=input_shape)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, nb_dim, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, nb_dim, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(nb_dim, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, nb_dim, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


def show_learning_graph(history, epochs, verbose=0):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def corrupted_img(path, apply_remove=False, verbose=0):
    num_skipped = 0
    to_remove = []
    for folder_name in get_sub_dir(path, verbose=verbose-1):
        folder_path = join(path, folder_name)
        for fname in listdir(folder_path):
            fpath = join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                to_remove.remove(fpath)
                # Delete corrupted image
                if apply_remove:
                    remove(fpath)

    print("Deleted %d images" % num_skipped)
    return to_remove


def get_sub_dir(dir_path, verbose=0):
    from glob import glob
    return glob(dir_path+ "/*/", recursive = True)


def get_dir_files(dir_path, endwith=None, include_sub_dir=0, verbose=0):

    fichiers = []

    if include_sub_dir > 0:
        first_level_sub_dir = get_sub_dir(dir_path, verbose=verbose-1)
        if len(first_level_sub_dir) > 0:
            for sub_dir in first_level_sub_dir:
                sub_f = get_dir_files(sub_dir, endwith=endwith, include_sub_dir=include_sub_dir-1, verbose=verbose)
                fichiers.extend([join(sub_dir, f) for f in sub_f])
        fichiers.extend(get_dir_files(dir_path, endwith=endwith, include_sub_dir=0, verbose=verbose))
    else:
        if endwith is not None:
            if isinstance(endwith, str):
                fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith(endwith)]
            elif isinstance(endwith, list):
                for en in endwith:
                    fichiers.extends(get_dir_files(dir_path=dir_path, endwith=en, verbose=verbose))
        else:
            fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return fichiers


def show_hog(img_path, reduce_ratio=None, cmap="BrBG",orientations=9, pixels_per_cell=(8, 8)):
    img = imread(img_path)
     
    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title(f'Original : {img.shape}')

    if reduce_ratio is not None:
        # resizing image
        resized_img = resize(img, (img.shape[0]/reduce_ratio, img.shape[1]/reduce_ratio))
    else:
        resized_img = img.copy()
    
    #creating hog features
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    ax3.axis('off')
    ax3.imshow(hog_image, cmap=cmap)
    ax3.set_title(f'Hog Image : {hog_image.shape}')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=cmap)
    ax2.set_title(f'Hog Rescaled : {hog_image_rescaled.shape}')
    plt.show()
    # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
    return img, resized_img, hog_image

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   

if __name__ == "__main__":
    
    print(get_dir_files(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\training_set', include_sub_dir=1, verbose=0))
    # print(get_dir_files(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset', include_sub_dir=1, verbose=0))
    print(get_sub_dir('dataset', verbose=0))

    # print(corrupted_img('dataset', apply_remove=False, verbose=0))

