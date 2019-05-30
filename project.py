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

# TODO: create tiled image from all 12 scans
# def tile_images(data):

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
    n,m,y,x = np.shape(data)
    for i in range(n):
        for j in range(m):
            data[i,j] = keras.applications.resnet50.preprocess_input(data[i,j]) # make sure the data is compatible with resnet

    all_data_input = np.zeros((n*m,y,x,3), dtype = np.float32) # make each scan its own input
    all_data_labels = np.zeros((n*m))
    for i in range(n):
        all_data_input[i*m:(i+1)*m] = np.repeat(data[i,:,:,:,np.newaxis], 3, axis=3) # replicate greyscale images to make 3 channels
        all_data_labels[i*m:(i+1)*m] = np.repeat(labels[i], m)

    # print(all_data_input.shape)
    # print(all_data_labels.shape)
    train_input, test_input, train_output, test_output = train_test_split(all_data_input, all_data_labels, test_size=0.2)
    train_output = to_categorical(train_output, num_classes=4)
    test_output = to_categorical(test_output, num_classes=4)
    return train_input, test_input, train_output, test_output

def tile_images(data):
    n,m,y,x = np.shape(data)
    output_images = np.zeros((n, 4*y, 3*x))
    for i in range(n):
        num_tiles = 12
        index = 0
        for j in range(4):
            for k in range(3):
                print(np.shape(output_images[i, j*y:(j+1)*y, k*x:(k+1)*x]))
                output_images[i, j*y:(j+1)*y, k*x:(k+1)*x] = data[i,index]
                index += 1
    return output_images

def main():
    data = util.load_processed_data(util.PREPROCESSED_DATA) # this is how we can load the data for conv nets
    n,m,y,x = np.shape(data)
    peri_vals = util.load_patient_labels(util.LABEL_DATA,"1","peri") #this outputs the average periventricular Fazekas score
    deep_vals = util.load_patient_labels(util.LABEL_DATA,"1","deep") #this outputs the average deep Fazekas score
    # util.multi_slice_subplot(data[1])
    # util.multi_slice_subplot(data[12])
    # util.multi_slice_subplot(data[40])
    # plt.show()

    # create input data from our data set
    train_input, test_input, train_output, test_output = get_data_split(data, peri_vals)

    # include non-empty weights path if you want to load pretrained model
    fazeka_model = build_model((y,x,3), trainable=False, weights_path="SavedWeights/weights_best.hdf5")
    fazeka_model.summary()

    # train the model
    fazeka_train = train_model(fazeka_model, train_input, test_input, train_output, test_output, 
        filepath="SavedWeights/weights_best", batch_size=10, epochs=5)

    # test the model
    K.set_learning_phase(0)
    test_eval = fazeka_model.evaluate(test_input, test_output, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])

if __name__ == '__main__':
    main()
