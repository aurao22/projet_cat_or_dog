from copy import deepcopy
from gc import callbacks
from os import listdir, remove, makedirs
from os.path import isfile, join, exists

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn
import cv2

from sklearn.metrics import confusion_matrix

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              MODELS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout


def make_model_cnn5(input_shape, num_classes, verbose=0):
    inputs = keras.Input(shape=input_shape)
    
    x = inputs

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
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
    outputs = layers.Dense(units, activation=activation, name=activation)(x)
    return keras.Model(inputs, outputs)



def make_model_cnn4(input_shape, num_classes, optimizer = Adam(learning_rate=0.000001), verbose=0):
    # Initialisation du modèle
    classifier = Sequential()

    # Réalisation des couches de Convolution  / Pooling

    # ---- Conv / Pool N°1
    classifier.add(Conv2D(filters=16,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        input_shape=input_shape,
                        activation='relu'))

    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # ---- Conv / Pool N°2
    classifier.add(Conv2D(filters=16,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))

    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # ---- Conv / Pool N°3
    classifier.add(Conv2D(filters=32,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))

    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # ---- Conv / Pool N°4
    classifier.add(Conv2D(filters=32,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))

    classifier.add(BatchNormalization())

    classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Fully Connected
    # Flattening : passage de matrices 3D vers un vecteur
    classifier.add(Flatten())
    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dropout(0.1))

    activation='softmax'
    units = num_classes
    loss='categorical_crossentropy'
    name='softmax'

    if num_classes == 2:
        activation = "sigmoid"
        units = 1
        loss = 'binary_crossentropy'
        name='sigmoid'
    
    # Couche de sortie : classification => softmax sur le nombre de classe
    classifier.add(Dense(
                    units=units,
                    activation=activation,
                    name=name))

    # compilation du model de classification
    classifier.compile(
        optimizer=optimizer,
        loss=loss,
        # loss,accuracy,val_loss,val_accuracy,lr
        metrics=['accuracy'])
    
    return classifier

def create_and_fit_cnn1(target_size,training_set, validation_set, epochs=5, model_path="model/cnn_v1_best.h5", verbose = 0):
    cnn=tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[target_size[0],target_size[1],3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

    cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
        
    #  Compilation et entrainement
    opt = Adam(learning_rate=0.000001)
    cnn.compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    
    # Entrainement
    save_call_back = keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                save_best_only=True,
                verbose=verbose)

    history_cnn = cnn.fit(training_set, epochs = epochs , validation_data=validation_set, callbacks=[save_call_back])
    if model_path is not None:
        mod_temp = model_path.replace("_best", "")
        mod_temp = mod_temp.replace(".h5", ".pkl")
        cnn.save(model_path.replace("_best", ""))
        if verbose:
            print(f"INFO : model saved : {mod_temp}")

    show_learning_graph(history=history_cnn, epochs=epochs, verbose=verbose)
    
    return cnn


def make_model_cnn2(input_shape, num_classes, data_augmentation=None, nb_dim=3, sizes=[128, 256, 512, 728, 1024], verbose=0):
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

    for size in sizes:
        if size != sizes[-1]:
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
        else:
            # On prend la dernière taille proposée
            x = layers.SeparableConv2D(size, nb_dim, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            break

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              PREDICTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def predict_img(model, img_test, target_size, labels,label_expected,verbose=0):
    img = keras.preprocessing.image.load_img(img_test, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)
    predict_class = int(predictions[0][0])
    label = labels[predict_class]

    if label == label_expected:
        return 1, predict_class
    else:
        if verbose>0:
            plt.figure(figsize=(5, 5))
            imread = cv2.imread(img_test)
            dst = cv2.cvtColor(imread, code=cv2.COLOR_BGR2RGB)
            plt.imshow(dst)
            plt.title(f"{label_expected} expected (predict_class : {predict_class}), {label} predict")
            plt.show()
        else:
            print(f"{label_expected} expected (predict_class : {predict_class}), {label} predict")
        return 0, predict_class

from copy import deepcopy

def predict_n_img(model, aurelie_test, aurelie_y,target_size, labels, verbose=0):
    success = 0
    fail = 0
    fail_files = []
    predictions = []

    for i in range(0, len(aurelie_test)):
        try:
            found, predict_class = predict_img(model=model,img_test=aurelie_test[i], target_size=target_size, labels=labels,label_expected=labels[aurelie_y[i]],verbose=verbose)
            predictions.append(predict_class)
            
            if found:
                success += 1
            else:
                fail += 1
                fail_files.append(aurelie_test[i])
        except Exception as e:
            print(i, aurelie_test[i], e)
            predictions.append(3)

    df_cm = confusion_matrix(aurelie_y, predictions)

    labels_temp = deepcopy(labels)
    # labels_temp.append('autre')

    # On ajoute les labels textuels pour cque ce soit plus lisible
    cm_array_df = pd.DataFrame(df_cm, index=labels_temp, columns=labels_temp)
    if verbose>0:
        print(cm_array_df)
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(cm_array_df, annot=True, annot_kws={"size": 12},fmt='g')
    return df_cm, fail_files 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              GRAPHES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training and Validation Loss')
    plt.show()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              LECTURE DES DATASETS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_aurelie_full_test(label_codes, verbose=0):

    aurelie_test = get_dir_files(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\aurelie_validation_set', include_sub_dir=1, verbose=verbose)
    aurelie_y = []

    for f in aurelie_test:
        category=f.split('\\')[-1]
        category=category.split('.')[0]
        aurelie_y.append(label_codes.get(category,0))
        
    return aurelie_test, aurelie_y 


def get_validation_img(label_codes, path=r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\validation_set', verbose=0):

    img_test = get_dir_files(path, include_sub_dir=1, verbose=verbose)
    img_test = sorted(img_test)
    img_y = []

    for f in img_test:
        category=f.split('\\')[-1]
        category=category.split('.')[0]
        img_y.append(label_codes.get(category,0))
        
    return img_test, img_y 

from skimage import io

def get_df_image(img_path_list, start_path, verbose=0):
    categories_names = []
    file_type = []
    img_height = []
    img_width = []
    img_dims = []
    img_datas = []

    for f_name in img_path_list:
        category=f_name.split('\\')[-1]
        category=category.split('.')[0]
        categories_names.append(category)
        file_type.append(f_name.split('.')[-1])
        coins_image = io.imread(join(start_path, f_name))
        img_h = np.nan
        img_w = np.nan
        img_dim = 3

        if coins_image is not None:
            img_w = coins_image.shape[0]
            img_h = coins_image.shape[1]
            if len(coins_image.shape)==3:
                img_dim = coins_image.shape[2]
        
        img_height.append(img_h)
        img_width.append(img_w)
        img_dims.append(img_dim)
            
    df=pd.DataFrame({
        'file_name':f_name,
        'file_type':file_type,
        'category_name':categories_names,
        'img_height':img_height,
        'img_width':img_width,
        'img_dim':img_dims,
    })
    return df

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TRAITEMENT DES IMAGES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def resize_and_replace_picture(img_path, dim=None, scale_percent=60, bigger_only=True, lower_only=False, verbose=0):
    img = cv2.imread(img_path)
    if img is not None:
        resized = resize_picture(img, dim=dim, scale_percent=scale_percent,bigger_only=bigger_only, lower_only=lower_only, verbose=verbose)
        if resized is not None:
            remove_file(img_path)
            cv2.imwrite(img_path, resized)
            return 1
    else:
        print(f"ERROR with : {img_path}")
    return 0

def resize_picture(img, dim=None, scale_percent=60, bigger_only=True, lower_only=False, verbose=0):
    
    if dim is not None:
        # Il faut calculer la dimension cible pour garder les proportions
        # calcul des 2 pourcentages
        width_scale = dim[1] / img.shape[1] * 100
        height_scale = dim[0] / img.shape[0] * 100
        scale_percent = width_scale if width_scale < height_scale else height_scale
    
    # Sinon on calcul la dimension cible par rapport au ratio
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    do_it = False

    if bigger_only:
        do_it = (img.shape[1] > dim[1] and  img.shape[0] > dim[0])

    if lower_only:
        do_it = (img.shape[1] < dim[1] and  img.shape[0] < dim[0])

    if do_it:
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized


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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              FILES UTILITIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from tensorflow import compat

def remove_file(file_path):
    try:
        if exists(file_path):
            return remove(file_path)
    except OSError as e:
        print(e)

def del_corrupt_img(dir_path, include_sub_dir=0, verbose=0):
    
    removed_files = []
    fichiers = get_dir_files(dir_path=dir_path, include_sub_dir=include_sub_dir, verbose=verbose-1)
        
    for fname in fichiers:
        is_jfif = False
        fpath = join(dir_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            removed_files.append(fpath)
            # Delete corrupted image
            remove(fpath)

    if verbose: print(f"Deleted {len(removed_files)} images.")
    return removed_files


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
            fichiers = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and not f.endswith("Thumbs.db")]
    return fichiers

from sklearn.model_selection import train_test_split
from shutil import copy2

def split_dataset_and_move_it(source_path, training_path, validation_path, validation_rate=0.2, random_state=0, verbose=0):
    # il faut lister tous les sous répertoire et les créés tel quel.
    sub_dirs = get_sub_dir(source_path, verbose=verbose)

    for sb in sub_dirs:
        sb_name = sb.replace(source_path, "")
        sb_name = sb_name.replace("\\", "")
        
        # Chargement de la liste d'images du sous dossier
        files = get_dir_files(sb, endwith=None, include_sub_dir=0, verbose=verbose-1)
        
        # split de la liste d'image
        train_files, test_files = train_test_split(files, test_size=validation_rate, random_state=random_state)
        to = join(training_path, sb_name)
        # on créé le répertoire cible s'il n'existe pas
        if not exists(to): makedirs(to)

        # copie des images dans les dossiers cibles
        for file in train_files:
            try:
                file_path = join(source_path,sb_name,file)
                copy2(file_path, to)
            except Exception as error:
                if verbose:
                    print(f"ERROR on {file} : {error}")

        to = join(validation_path, sb_name)
        # on créé le répertoire cible s'il n'existe pas
        if not exists(to): makedirs(to)
        
        for file in test_files:
            try:
                file_path = join(source_path,sb_name,file)
                copy2(file_path, to)
            except Exception as error:
                if verbose:
                    print(f"ERROR on {file} : {error}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              IMG UTILITIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def move_n_files(source, destination, rate):
    pass
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   

if __name__ == "__main__":
    
    removed_files = del_corrupt_img(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\training_set', include_sub_dir=1, verbose=1)
    removed_files = del_corrupt_img(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\validation_set', include_sub_dir=1, verbose=1)


    print(get_dir_files(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset\training_set', include_sub_dir=1, verbose=0))
    # print(get_dir_files(r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_cat_or_dog\dataset', include_sub_dir=1, verbose=0))
    print(get_sub_dir('dataset', verbose=0))

    # print(corrupted_img('dataset', apply_remove=False, verbose=0))

