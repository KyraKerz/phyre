import random
import keras
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, LeakyReLU, MaxPooling2D
import numpy as np
import animations
import tensorflow as tf
# from keras_video import VideoFrameGenerator
from matplotlib import pyplot as plt
from keras.layers.merge import Concatenate
import random_agents as ra
from phyre import SimulationStatus
import PrepareSamples as ps
import phyre
from keras.layers.merge import Concatenate
from tensorflow.python.layers import layers


def train_model(numberOfEpochs,batch_size,is_update=False):
    random.seed(0)
    images, action, label = ps.prepareSamplesWithScore(is_update)
    imsize = images[0].shape[0]
    print(imsize)
    imagesnew = images  # images.reshape(images.shape[0], imsize, imsize, 1)
    images = images  # /np.max(images)
    # imagesnew = images[0::2]

    length = int(0.7 * label.shape[0])
    print(length)
    end = int(0.3 * label.shape[0])


    images_train = imagesnew[0:length]
    actions_train = action[0:length]
    label_train = label[0:length]
    # label_train = keras.utils.to_categorical(label[0:length], num_classes=2, dtype='int64')
    # evaluation_train = keras.utils.to_categorical(label_train, num_classes=2, dtype='int64')
    images_validation = imagesnew[length:length + end]
    actions_validation = action[length:length + end]
    label_validation = label[length:length + end]
    # label_validation = keras.utils.to_categorical(label[length:length+end], num_classes=2, dtype='int64')
    images_test = imagesnew[length + end:]
    label_test = label[length + end:]
    # label_test = keras.utils.to_categorical(label[length+end:], num_classes=2, dtype=int)
    print(images_test.shape[0])

    input_img = keras.Input(shape=(imsize, imsize, 3))
    vector_input = keras.Input((3,))
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

    x = keras.layers.Conv2D(imsize, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
    x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(imsize, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    flatten = keras.layers.Flatten()(x)  # append three no: x,z radius
    # concat_layer= flatten#Concatenate()([vector_input, flatten])
    concatenate = keras.layers.concatenate([flatten, vector_input])
    dense1 = keras.layers.Dense(imsize * 2, activation='relu', kernel_initializer=initializer)(concatenate)
    dense2 = keras.layers.Dense(1, activation='linear')(dense1)

    # model = models.Sequential()
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(imsize, imsize, 3)))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dense(2, activation = 'softmax'))

    # model = keras.Model(inputs= input_img, outputs = dense2)
    model = keras.Model(inputs=[input_img, vector_input], outputs=dense2)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit([images_train, actions_train], label_train, batch_size=batch_size, epochs=numberOfEpochs
                        , verbose=1, validation_data=([images_validation, actions_validation], label_validation))

    model.save('model2')

    # history = model.fit([images_train, actions_train], evaluation_train, batch_size=2,epochs=30,verbose=1,validation_data=([images_validation, actions_validation], evaluation_validation))
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss cnn2')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


# train_model()
