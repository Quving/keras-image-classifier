![alt text](https://viblo.asia/uploads/1b042898-d4d8-4a90-b7aa-831eea3a5f83.png)


# keras-image-classifier
This repository contains different subprojects with different approaches to train a classifier. In scope of a master project, the task was to classify different wooden blocks of different color and shapes. Unlike usual blogs and github repositories, this repository provide the full code as well as the used training samples. The training samples are recorded by myself and they can be located [here](https://github.com/Quving/keras-image-classifier/tree/master/data/). Enjoy!

___

## Install dependencies
``` pip install -r requirements.txt ```
___
## Data set
Download the samples the neural network was trained with can be obtained with following command.

``` wget -O original_samples.zip http://nextcloud.quving.com/s/5B4zfnLSMXsa37R/download ```
### Training samples
As mentioned in the description, the data are collected by myself using the [xbox one kinect](https://www.xbox.com/de-DE/xbox-one/accessories/kinect). The samples have been collected on a white table. Hence, they are quite idealistic for classification since they are not embedded in a cluttered environment. With this background, it is sufficient to record ~130 samples of each class to maintain a very high accuracy (~99%). In order to prevent overfitting, but to gain robustness, the training samples are augmented via a number of random transformations. In the end, each class are represented by **1000 samples**.

#### Validation samples
Each class has **200 validation samples**. They have been splitted up from the bunch of augmented samples.

#### Test samples
Test samples are not involved in the training stage. They have been created for you to test the neural network if you don't believe in my working code snippets.

### Using a pre-trained network (here VGG16) (Achieved accuracy ~ 99%)
[use_vgg16/](https://github.com/Quving/keras-image-classifier/tree/master/use-vgg16)

___

## Usage

### Before training:

``` bash initialize.sh ```

##### Please keep in mind
This repository does not provide the trained weights of the neural networks. Once the script's are executed, the model's structure and weights are exported as default into the the folder **models/**.

## Documentations
[Keras](https://keras.io/)

___
## References
- [building-powerful-image-classification-models-using-very-little-data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [bottleneck-features-multi-class-classification-keras](http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html)
