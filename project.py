import util

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras
import nibabel as nib
import scipy as scp

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import keras.backend as K

# This project is available at https://github.com/sbauersfeld/CS168Project
# Authors: Scott Bauersfeld, Serene Kamal, Meena Nagappan

# The idea of transfer learning as used here is based on the tensorflow tutorial "Transfer Learning Using Pretrained ConvNets"
# it is publicly available online at: https://www.tensorflow.org/alpha/tutorials/images/transfer_learning 
def build_model(img_shape, trainable=False, fine_tune_at=165, weights_path=""):
    """
    Build the CNN architecture with a pre-trained ResNet50 base layer.
        
    Parameters
    --------------------
        image shape    -- shape of inputs that will be used
        trainable      -- whether or not the pre-trained layer can be re-trained
        fine_tune_at   -- the layer number to begin re-training at, only if trainable is True
        weights_path   -- a path to saved weights that will be loaded onto the entire architecture if non-empty
                          use this to load previously saved models
    
    Returns
    --------------------
        fazeka model    -- the CNN model used for our project
    """

    # get the pre-trained ResNet50
    base_model = keras.applications.ResNet50(input_shape=img_shape,include_top=False, weights='imagenet')
    base_model.trainable = trainable

    # Freeze all the layers before the fine_tune_at layer
    if trainable:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

    # add our own classification layers
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    d_layer = keras.layers.Dense(1024, activation='relu')
    dropout = keras.layers.Dropout(0.5) # large dropout to combat overfitting
    prediction_layer = keras.layers.Dense(4, activation='softmax')

    # create our model
    fazeka_model = keras.Sequential([base_model, global_average_layer, d_layer, dropout, prediction_layer])
    
    if weights_path != "":  # load the saved weights, if any are specified
        fazeka_model.load_weights(weights_path)

    # compile model with categorical_crossentropy loss, Adam optimizer
    fazeka_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return fazeka_model

def train_model(model, train_input, test_input, train_output, test_output, filepath, batch_size=10, epochs=35):
    """
    Train the model.
        
    Parameters
    --------------------
        the model to train 
        train/test input
        train/test labels
        filepath specifying where to save weights
        batch size for training
        number of epochs to train for
    
    Returns
    --------------------
        fazeka train    -- the results of training
    """
    K.set_learning_phase(1)
    # set the location to save weights during training, only save the weights producing the highest accuracy
    checkpoint = ModelCheckpoint(filepath, save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # train the model
    fazeka_train = model.fit(train_input, train_output, callbacks=callbacks_list, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_input, test_output))
    return fazeka_train

def get_data_split(data, labels):
    """
    Split the data into training and testing sets.
        
    Parameters
    --------------------
        data    -- the data to split
        labels  -- the labels to split
    
    Returns
    --------------------
        training/testing inputs
        training/testing labels
        indices of original data used for training
        indices of original data used for testing
    """

    indices = np.arange(len(data))
    # get train/test split
    train_input, test_input, train_output, test_output, train_idx, test_idx = train_test_split(data, labels, indices, stratify=labels, test_size=0.15)
    print("Training with samples: ", train_idx)
    print("Testing with samples: ", test_idx)
    return train_input, test_input, train_output, test_output, train_idx, test_idx


def tile_images(data, labels):
    """
    Convert the 3D MRIs into 2D tiled images
        
    Parameters
    --------------------
        data    -- the 3D MRI scans
        labels  -- the labels for each patient
    
    Returns
    --------------------
        output images   -- the tiled 2D images
        labels          -- the labels corresponding to each output image
    """
    n,m,y,x = np.shape(data)
    output_images = np.zeros((n, 4*y, 3*x))
    for i in range(n):
        index = 0
        for j in range(4):
            for k in range(3): # tile each of the 12 slices onto a 2D image
                output_images[i, j*y:(j+1)*y, k*x:(k+1)*x] = data[i,index]
                index += 1

    output_images = np.repeat(output_images[:,:,:,np.newaxis], 3, axis=3) # replicate each image twice to make it three channels
    return output_images, labels

def unravel_scans(data, labels):
    """
    Convert the 3D MRIs into individual scans.
        
    Parameters
    --------------------
        data    -- the 3D MRI scans
        labels  -- the labels for each patient
    
    Returns
    --------------------
        output images   -- the unraveled scans
        labels          -- the labels corresponding to each output image
    """

    n,m,y,x = np.shape(data)
    all_data_input = np.zeros((n*m,y,x,3), dtype = np.float32) # make each scan its own input
    all_data_labels = np.zeros((n*m))
    for i in range(n):  # transform 3D data into a list of 2D images
        all_data_input[i*m:(i+1)*m] = np.repeat(data[i,:,:,:,np.newaxis], 3, axis=3) # replicate greyscale images to make 3 channels
        all_data_labels[i*m:(i+1)*m] = np.repeat(labels[i], m)

    return all_data_input, all_data_labels

def build_train_wrapper(data, original_labels, transform_function, load_weights_path="", save_weights_path="", batch_size=10, epochs=20):
    """
    A wrapper function to build, train, and test the Fazeka CNN.
        
    Parameters
    --------------------
        data                -- the data to train/test on 
        original labels     -- the labels for each sample in the data set
        transform function  -- either tile_images or unravel_scans
        load weights path   -- the path to load pre-saved weights from, if non-empty
        save weights path   -- the path to save weights to during training
        batch size          -- the number of samples to use during a batch when training
        epochs              -- the number of epochs to train for
    
    Returns
    --------------------
        fazeka model    -- the trained fazeka model
        train idx       -- the indices of the data that were chosen to be used as training data
        test idx        -- the indices of the data that were chosen to be used as testing data
    """

    # be sure to split the data by patient, so that unravel scans approach is not misused
    train_input, test_input, train_output, test_output, train_idx, test_idx = get_data_split(data, original_labels)
    
    # transform 3D data to 2D
    train_input, train_output = transform_function(train_input, train_output)
    test_input, test_output = transform_function(test_input, test_output)
    n,y,x,z = np.shape(train_input)

    # format categorical labels
    train_output = to_categorical(train_output, num_classes=4)
    test_output = to_categorical(test_output, num_classes=4)

    # include non-empty weights path if you want to load our pretrained model
    fazeka_model = build_model((y,x,3), trainable=False, weights_path=load_weights_path)
    fazeka_model.summary()

    if save_weights_path == "":
        ext = "_" + datetime.datetime.now().strftime("%m%d-%H%M%S") + ".hdf5"
        if transform_function == unravel_scans:
            save_weights_path == "SavedWeights/unravel_weights_best"
        elif transform_function == tile_images:
            save_weights_path = "SavedWeights/tile_weights_best"
        save_weights_path += ext

    print("Saving weights to file:", save_weights_path)

    # train the model
    fazeka_train = train_model(fazeka_model, train_input, test_input, train_output, test_output, 
        filepath=save_weights_path, batch_size=batch_size, epochs=epochs)

    # plot accuracy over time
    plt.plot(fazeka_train.history['acc'])
    plt.plot(fazeka_train.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # plot loss over time
    plt.plot(fazeka_train.history['loss'])
    plt.plot(fazeka_train.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # test the model
    K.set_learning_phase(0)
    test_eval = fazeka_model.evaluate(test_input, test_output, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    # inspect the results by comparing predicted scores to true scores and displaying the test data
    predicted_scores = fazeka_model.predict_classes(test_input)
    for i in range(len(test_idx)):
        if transform_function == unravel_scans:
            idx_l = i * 12
            idx_h = (i+1) * 12  
            print("Predicted scores for patient:")
            print(predicted_scores[idx_l:idx_h])
            print("True scores:")
            print(np.argmax(test_output[idx_l:idx_h]).astype(int))
            util.multi_slice_subplot(test_input[idx_l:idx_h,:,:,0])
        elif transform_function == tile_images:
            print("Predicted scores for patient:")
            print(predicted_scores[i])
            print("True scores:")
            print(np.argmax(test_output[i]).astype(int))
            plt.figure(i)
            plt.imshow(test_input[i,:,:,0], cmap='gray')
    plt.show()
    return fazeka_model, train_idx, test_idx

def evaluate_results(test_input, test_output, model_method, weights_path):
    """
    A function to evaluate a previously trained model.
        
    Parameters
    --------------------
        test input      -- the data to test on
        test output     -- the true labels for the test data
        model method    -- either tile_images or unravel_scans
        weights path    -- the path to load the previously saved weights from
    """
    L = len(test_input)

    # transform 3D -> 2D
    test_input, test_output = model_method(test_input, test_output)
    n,y,x,z = np.shape(test_input)
    cat_output = to_categorical(test_output, num_classes=4)

    # build the model from saved weights
    fazeka_model = build_model((y,x,3), trainable=False, weights_path=weights_path)

    # test the model
    K.set_learning_phase(0)
    test_eval = fazeka_model.evaluate(test_input, cat_output, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

    # inspect the results
    predicted_scores = fazeka_model.predict_classes(test_input)
    for i in range(L):
        if model_method == unravel_scans:
            idx_l = i * 12
            idx_h = (i+1) * 12  
            print("Predicted scores for patient:")
            print(predicted_scores[idx_l:idx_h])
            print("True scores:")
            print(test_output[idx_l:idx_h].astype(int))
            util.multi_slice_subplot(test_input[idx_l:idx_h,:,:,0])
        elif model_method == tile_images:
            print("Predicted scores for patient:")
            print(predicted_scores[i])
            print("True scores:")
            print(test_output[i].astype(int))
            plt.figure(i)
            plt.imshow(test_input[i,:,:,0], cmap='gray')
    plt.show()

def main():
    data = util.load_processed_data(util.PREPROCESSED_Z_SCORES) # this is how we can load the data for conv nets
    peri_vals = util.load_patient_labels(util.LABEL_DATA,"1","peri") #this outputs the average periventricular Fazekas score
    deep_vals = util.load_patient_labels(util.LABEL_DATA,"1","deep") #this outputs the average deep Fazekas score
    # util.multi_slice_subplot(data[1]) # this is how you can see the data
    # util.multi_slice_subplot(data[12])

    load_weights_path = "" # can load pre-saved weights

    # train/load the model
    labels = deep_vals                         # choose the labels to use here

    # remove patient 40 because they could not be registered properly
    data = np.delete(data, 40, 0)
    labels = np.delete(labels, 40, 0)
    model_input_param = "tile_deep_re1"        # name the save file
    model_method = tile_images                 # choose the model method (tile_images or unravel_scans)
    batch_size = 5                             # probably set this to 5 for tile images, 10 for unravel
    epochs = 20                                # probably set this to 10 for tile images, 20 for unravel
    time = "_" + datetime.datetime.now().strftime("%m%d-%H%M")
    save_weights_path = "SavedWeights/" + model_input_param + time + ".hdf5"

    # build, train, and test a fazeka CNN
    fazeka_model, train_idx, test_idx = build_train_wrapper(data, labels, model_method, 
        load_weights_path="", save_weights_path=save_weights_path, 
        batch_size=batch_size, epochs=epochs)

    # save the indices used for testing and training
    save_test_data_inputs_path = "TestInputs/" + model_input_param + "_input" + time
    mat_dict = {}
    mat_dict['train'] = train_idx
    mat_dict['test'] = test_idx
    scp.io.savemat(save_test_data_inputs_path, mat_dict) # save test/train data   

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()
