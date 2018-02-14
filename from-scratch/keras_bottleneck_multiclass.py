import os
import cv2
import math
import json
import pickle
import numpy as np
# import matplotlib.pyplot as plt

from keras import backend as K
from keras import optimizers
from keras import applications
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# dimensions of our images.
img_width, img_height = 224, 224

# set paths
train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
top_model_path = 'models/top_model_17_classes'
history_path = 'training/training_history_17_classes.json'
class_indices_path = 'class_indices/class_indices_17_classes.npy'
bottleneck_features_train_samples = 'bottleneck_features/bottleneck_features_train_samples.npy'
bottleneck_features_validation_samples = 'bottleneck_features/bottleneck_features_validation_samples.npy'

# train parameters.
epochs = 50
batch_size = 20
learning_rate = 0.0001

# Persist history of training stage.
def save_history(history, filename):
    print "Persist training history in", filename
    with open(filename, 'wb') as outfile:
        pickle.dump(history.history, outfile)
        outfile.close()

# Persist model structure and its weights.
def save_model(model, filename):
    print "Persist model completely in", filename
    with open(filename + '.json', 'w') as outfile:
        outfile.write(model.to_json(sort_keys=True,
            indent=4,
            separators=(',', ': ')))
        outfile.close()

     # Save weights
    model.save(filename + '.h5')

def train_top_model():
    # load the bottleneck features saved earlier
    train_data = np.load(bottleneck_features_train_samples)

    # get the class lebels for the training data, in the original order
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    training_gen = datagen_top.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    validation_gen = datagen_top.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

    nb_train_samples = len(training_gen.filenames)
    nb_validation_samples = len(validation_gen.filenames)
    num_classes = len(training_gen.class_indices)
    print training_gen.class_indices

    # save the class indices to use use later in predictions
    np.save(class_indices_path, training_gen.class_indices)
    model = create_top_model(train_data.shape[1:], num_classes)

    # Training stage.
    history = model.fit_generator(
            training_gen,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_gen,
            validation_steps=nb_validation_samples//batch_size)

    # Persist history and model.
    save_history(history, history_path)
    save_model(model, top_model_path)
    validation_labels = validation_gen.classes
    validation_labels = to_categorical(
            validation_labels, num_classes=num_classes)
    (eval_loss, eval_accuracy) = model.evaluate_generator( validation_gen, nb_validation_samples/batch_size)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))


def plot_loss(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def create_top_model(shape, num_classes):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # build top model
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_top_model()

