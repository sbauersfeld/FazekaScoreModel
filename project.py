import util

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import keras

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
    filepath += "_" + datetime.datetime.now().strftime("%m%d-%H%M%S") + ".hdf5"
    print("Saving weights to file:", filepath)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    fazeka_train = model.fit(train_input, train_output, callbacks=callbacks_list, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_input, test_output))
    return fazeka_train

def get_data_split(data, labels):
    # n,m,y,x = np.shape(data)
    # for i in range(n):
    #     for j in range(m):
            # data[i,j] = keras.applications.resnet50.preprocess_input(data[i,j]) # make sure the data is compatible with resnet
    data = keras.applications.resnet50.preprocess_input(data)

    # print(all_data_input.shape)
    # print(all_data_labels.shape)
    train_input, test_input, train_output, test_output = train_test_split(data, labels, test_size=0.2)
    train_output = to_categorical(train_output, num_classes=4)
    test_output = to_categorical(test_output, num_classes=4)
    return train_input, test_input, train_output, test_output

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
    input_data, labels = transform_function(data, original_labels)
    n,y,x,z = np.shape(input_data)
    print(np.shape(input_data))

    train_input, test_input, train_output, test_output = get_data_split(input_data, labels)

    # include non-empty weights path if you want to load our pretrained model
    fazeka_model = build_model((y,x,3), trainable=False, weights_path=load_weights_path)
    fazeka_model.summary()

    # train the model
    if save_weights_path == "":
        if transform_function == unravel_scans:
            save_weights_path == "SavedWeights/unravel_weights_best"
        elif transform_function == tile_images:
            save_weights_path = "SavedWeights/tile_weights_best"

    fazeka_train = train_model(fazeka_model, train_input, test_input, train_output, test_output, 
        filepath=save_weights_path, batch_size=batch_size, epochs=epochs)

    return fazeka_model, test_input, test_output

def main():
    data = util.load_processed_data(util.PREPROCESSED_Z_SCORES) # this is how we can load the data for conv nets
    peri_vals = util.load_patient_labels(util.LABEL_DATA,"1","peri") #this outputs the average periventricular Fazekas score
    deep_vals = util.load_patient_labels(util.LABEL_DATA,"1","deep") #this outputs the average deep Fazekas score
    # util.multi_slice_subplot(data[1])
    # util.multi_slice_subplot(data[12])
    # util.multi_slice_subplot(data[40])
    # plt.show()

    # test the model
    # fazeka_model, test_input, test_output = build_train_wrapper(data, peri_vals, tile_images, 
    #     load_weights_path="", save_weights_path="", batch_size=5, epochs=20)
    
    # K.set_learning_phase(0)
    # test_eval = fazeka_model.evaluate(test_input, test_output, verbose=0)
    # print('Test loss:', test_eval[0])
    # print('Test accuracy:', test_eval[1])

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
