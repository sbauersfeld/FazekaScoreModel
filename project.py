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

# TODO: normalize images
# def normalize(data):
# consider tf.image.per_image_standardization(image)

# TODO: create multiple input model

# TODO: visualize results, training + testing accuracy, etc

def build_model(img_shape, trainable=False, fine_tune_at=170, weights_path=""):
    base_model = keras.applications.ResNet50(input_shape=img_shape,include_top=False, weights='imagenet')
    base_model.trainable = trainable    
    # print("Number of layers in the base model: ", len(base_model.layers))

    # Freeze all the layers before the fine_tune_at layer
    if trainable:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

    # base_model.summary()

    global_average_layer = keras.layers.GlobalAveragePooling2D()
    d_layer = keras.layers.Dense(1024, activation='relu')
    dropout = keras.layers.Dropout(0.5)
    prediction_layer = keras.layers.Dense(4, activation='softmax')

    fazeka_model = keras.Sequential([base_model, global_average_layer, d_layer, dropout, prediction_layer])
    
    if weights_path != "":
        fazeka_model.load_weights(weights_path)
    fazeka_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    return fazeka_model

def train_model(model, train_input, test_input, train_output, test_output, filepath, batch_size=10, epochs=35):
    K.set_learning_phase(1)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    fazeka_train = model.fit(train_input, train_output, callbacks=callbacks_list, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_input, test_output))
    return fazeka_train

def get_data_split(data, labels):
    data = keras.applications.resnet50.preprocess_input(data)

    # print(all_data_input.shape)
    # print(all_data_labels.shape)
    indices = np.arange(len(data))
    train_input, test_input, train_output, test_output, train_idx, test_idx = train_test_split(data, labels, indices, test_size=0.15)
    print("Training with samples: ", train_idx)
    print("Testing with samples: ", test_idx)
    return train_input, test_input, train_output, test_output, train_idx, test_idx

def tile_images(data, labels):
    n,m,y,x = np.shape(data)
    output_images = np.zeros((n, 4*y, 3*x))
    for i in range(n):
        index = 0
        for j in range(4):
            for k in range(3):
                output_images[i, j*y:(j+1)*y, k*x:(k+1)*x] = data[i,index]
                index += 1

    output_images = np.repeat(output_images[:,:,:,np.newaxis], 3, axis=3)
    return output_images, labels

def unravel_scans(data, labels):
    n,m,y,x = np.shape(data)
    all_data_input = np.zeros((n*m,y,x,3), dtype = np.float32) # make each scan its own input
    all_data_labels = np.zeros((n*m))
    for i in range(n):
        all_data_input[i*m:(i+1)*m] = np.repeat(data[i,:,:,:,np.newaxis], 3, axis=3) # replicate greyscale images to make 3 channels
        all_data_labels[i*m:(i+1)*m] = np.repeat(labels[i], m)

    return all_data_input, all_data_labels

def build_train_wrapper(data, original_labels, transform_function, load_weights_path="", save_weights_path="", batch_size=10, epochs=20):
    # input_data, labels = transform_function(data, original_labels)
    # n,y,x,z = np.shape(input_data)
    # print("Input data shape: ", np.shape(input_data))
    # train_input, test_input, train_output, test_output = get_data_split(input_data, labels)

    train_input, test_input, train_output, test_output, train_idx, test_idx = get_data_split(data, original_labels)
    train_input, train_output = transform_function(train_input, train_output)
    test_input, test_output = transform_function(test_input, test_output)
    n,y,x,z = np.shape(train_input)

    train_output = to_categorical(train_output, num_classes=4)
    test_output = to_categorical(test_output, num_classes=4)

    # include non-empty weights path if you want to load our pretrained model
    fazeka_model = build_model((y,x,3), trainable=False, weights_path=load_weights_path)
    fazeka_model.summary()

    # train the model
    if save_weights_path == "":
        ext = "_" + datetime.datetime.now().strftime("%m%d-%H%M%S") + ".hdf5"
        if transform_function == unravel_scans:
            save_weights_path == "SavedWeights/unravel_weights_best"
        elif transform_function == tile_images:
            save_weights_path = "SavedWeights/tile_weights_best"
        save_weights_path += ext

    print("Saving weights to file:", save_weights_path)
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

    plt.plot(fazeka_train.history['loss'])
    plt.plot(fazeka_train.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    return fazeka_model, train_idx, test_idx

def evaluate_results(test_input, test_output, model_method, weights_path):
    L = len(test_input)
    test_input, test_output = model_method(test_input, test_output)
    n,y,x,z = np.shape(test_input)
    cat_output = to_categorical(test_output, num_classes=4)
    fazeka_model = build_model((y,x,3), trainable=False, weights_path=weights_path)

    test_input = keras.applications.resnet50.preprocess_input(test_input)
    K.set_learning_phase(0)
    test_eval = fazeka_model.evaluate(test_input, cat_output, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

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
    data = util.load_processed_data(util.PREPROCESSED_DATA) # this is how we can load the data for conv nets
    peri_vals = util.load_patient_labels(util.LABEL_DATA,"1","peri") #this outputs the average periventricular Fazekas score
    deep_vals = util.load_patient_labels(util.LABEL_DATA,"1","deep") #this outputs the average deep Fazekas score
    # util.multi_slice_subplot(data[1])
    # util.multi_slice_subplot(data[12])
    # util.multi_slice_subplot(data[40])
    # plt.show()
    # load_weights_path = "" # can load pre-saved weights


    # train/load the model
    labels = peri_vals                          # choose the labels to use here
    model_input_param = "unravel_peri_re1"      # name the save file
    model_method = unravel_scans                # choose the model method (tile or unravel)
    batch_size = 10                             # probably set this to 5 for tile images, 10 for unravel
    epochs = 30                                 # probably set this to 10 for tile images, 20 for unravel
    time = "_" + datetime.datetime.now().strftime("%m%d-%H%M")
    save_weights_path = "SavedWeights/" + model_input_param + time + ".hdf5"

    fazeka_model, train_idx, test_idx = build_train_wrapper(data, labels, model_method, 
        load_weights_path="", save_weights_path=save_weights_path, 
        batch_size=batch_size, epochs=epochs)

    # save which data was used for training/testing
    save_test_data_inputs_path = "TestInputs/" + model_input_param + "_input" + time
    mat_dict = {}
    mat_dict['train'] = train_idx
    mat_dict['test'] = test_idx
    scp.io.savemat(save_test_data_inputs_path, mat_dict)

    # save_test_data_inputs_path = "TestInputs/test_tile_peri_re1_input_0531-1934.mat"
    # inputs = scp.io.loadmat(save_test_data_inputs_path)
    # train_idx = inputs['train'][0]
    # test_idx = inputs['test'][0]
    # print(test_idx)
    # save_weights_path = "SavedWeights/test_tile_peri_re1_0531-1934.hdf5"
    
    test_input, test_output = data[test_idx], labels[test_idx] 
    evaluate_results(test_input, test_output, model_method, save_weights_path)   

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()
