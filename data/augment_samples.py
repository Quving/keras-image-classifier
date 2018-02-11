#!/usr/bin/python3.6

import os
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# From a given path the last name of the golder will be returned.
def get_name(source_path):
    arr = source_dir.split("/")
    index = len(arr)
    index -= 2 if source_dir.endswith("/") else 1
    return arr[index]


def generate_data(source_dir, target_dir, number):
    datagen = ImageDataGenerator(
            width_shift_range=0.075,
            height_shift_range=0.075,
            shear_range=0.075,
            zoom_range=0.075,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest')

    # Count generated samples.
    no_of_generated_samples = 0

    # Collect a list of names of the source directory.

    files = [f for f in os.listdir(source_dir) if ".jpg" in f]
    print(len(files), "images found in", source_dir)

    for imagefile in files:
        imagefile = source_dir+ "/" + imagefile

        img = load_img(imagefile)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # and saves the results to the `preview/` directory
        for batch in datagen.flow(x, batch_size=1,
                save_to_dir=target_dir,
                save_prefix=get_name(target_dir),
                save_format="jpg"):
            no_of_generated_samples += 1

            if no_of_generated_samples % int(number) == 0:
                break  # otherwise the generator would loop indefinitely

    print "=== SUMMARY ==="
    print "    ", len(files), "has been found."
    print "    ", no_of_generated_samples, "has been generated and has been saved to", target_dir


if __name__ == "__main__":
    # Parse script arguments.
    if not len(sys.argv) == 4:
        print("Please pass arguments.")
        print("1 source dir")
        print("2 target dir")
        print("3 number to be generated")
        exit()
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    number = sys.argv[3]
    generate_data(source_dir, target_dir, number)
