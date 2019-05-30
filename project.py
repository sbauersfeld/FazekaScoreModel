import util

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# TODO: normalize images
# def normalize(data):
# consider tf.image.per_image_standardization(image)

# TODO: create tiled image from all 12 scans
# def tile_images(data):

# TODO: create multiple input model

# TODO: visualize results, training + testing accuracy, etc

def main():
    data = util.load_processed_data(util.PREPROCESSED_DATA) # this is how we can load the data for conv nets
    n,m,y,x = np.shape(data)
    peri_vals = util.load_patient_labels(util.LABEL_DATA,"1","peri") #this outputs the average periventricular Fazekas score
    deep_vals = util.load_patient_labels(util.LABEL_DATA,"1","deep") #this outputs the average deep Fazekas score
    print(np.shape(peri_vals))
    print(np.shape(data))
    # util.multi_slice_subplot(data[1])
    # util.multi_slice_subplot(data[12])
    # util.multi_slice_subplot(data[40])
    # plt.show()

    # note that we will need to clone the grayscale images 2 times (for 3 channels) in order to use pretrained model
    base_model = keras.applications.xception.Xception(input_shape=(y,x,3),include_top=False, weights='imagenet')
    # base_model.trainable = False
    print("Number of layers in the base model: ", len(base_model.layers))
    # Fine tune from this layer onwards
    fine_tune_at = 128

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    # base_model.summary()

    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(4, activation='softmax')

    fazeka_model = keras.Sequential([base_model, global_average_layer, prediction_layer])
    fazeka_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    fazeka_model.summary()


    all_data_input = np.zeros((n*m,y,x,3), dtype = np.float32) # make each scan its own input
    all_data_labels = np.zeros((n*m))
    for i in range(n):
        all_data_input[i*m:(i+1)*m] = np.repeat(data[i,:,:,:,np.newaxis], 3, axis=3) # replicate greyscale images to make 3 channels
        all_data_labels[i*m:(i+1)*m] = np.repeat(peri_vals[i], m)

    print(all_data_input.shape)
    print(all_data_labels.shape)
    train_input, test_input, train_output, test_output = train_test_split(all_data_input, all_data_labels, test_size=0.2)
    train_output = to_categorical(train_output, num_classes=4)
    test_output = to_categorical(test_output, num_classes=4)

    batch_size = 10
    epochs = 20
    fazeka_train = fazeka_model.fit(train_input, train_output, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_input, test_output))
    test_eval = fazeka_model.evaluate(test_input, test_output, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1]) 

    fazeka_model.save('test1.h5')

if __name__ == '__main__':
    main()
