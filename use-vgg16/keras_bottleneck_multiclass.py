import os
import cv2
import math
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras import optimizers
from keras import applications
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
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


# Extract the bottleneck features of the training and validation samples.
def save_bottlebeck_features():
    model = applications.VGG16(include_top=False, weights='imagenet') # build the VGG16 network

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    # Number of training samples and Number of classes.
    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
            generator, predict_size_train)

    np.save(bottleneck_features_train_samples, bottleneck_features_train)

    generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    # Number of validation samples.
    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
            math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
            generator, predict_size_validation)

    np.save(bottleneck_features_validation_samples, bottleneck_features_validation)


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    print generator_top.class_indices

    # save the class indices to use use later in predictions
    np.save(class_indices_path, generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load(bottleneck_features_train_samples)

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load(bottleneck_features_validation_samples)

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
            validation_labels, num_classes=num_classes)

    model = create_top_model(train_data.shape[1:], num_classes)
        # Training stage.
    history = model.fit(train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(validation_data, validation_labels))

    # Persist history and model.
    save_history(history, history_path)
    plot_loss(history)
    save_model(model, top_model_path)

    (eval_loss, eval_accuracy) = model.evaluate(
            validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))


def plot_loss(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def create_top_model(shape, num_classes):
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy', metrics=['accuracy'])
    return model



if __name__ == "__main__":
    save_bottlebeck_features()
    train_top_model()
    history = pickle.load( open(history_path,"rb"))
    plot_loss(history)
