# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/original.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128. The network architecture is modified from nvidia's paper. Since I don't have as much data as they do, I use a 500 depth convolution layer to replace the 1000 dense layer to avoid overfitting. I also add some dropout layers. There are two batch normalization layers so that the gradient can get back to the first several layers.


#### 2. Attempts to reduce overfitting in the model

As described above, the model contains dropout layers and a 1X1 convolution layer(to reduce the amount of parameters) in order to reduce overfitting. I also use some data augments by flipping the image, use side camera images.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, reverse driving and driving on the second track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy the nvidia architecture and modified it to suit the problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to add some dropout layers.

Then I add a 1x1 500 filters convolution layer and I found the training get stuck so I then add two batch norm layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track when there was a sharp turn. To improve the driving behavior in these cases, I drove around those turns several times to collect more data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3  image                            | 
| Cropping              | Cut the top 60 and bottom 25 pixels       |
| Lambda Normalization  |         Normalize the  input             |
| Convolution 5x5     | 2x2 stride, valid padding, activation relu, output 24|
| Convolution 5x5     | 2x2 stride, valid padding, activation relu, output 36|
| Convolution 5x5   | 2x2 stride, valid padding, activation relu, output 48|
| Batch Normalization   |                                                 |
| Convolution 3x3      | 1x1 stride, valid padding, activation relu, output 64|
| Convolution 3x3    | 1x1 stride, valid padding, activation relu, output 64 |
| Convolution 1x1    | 1x1 stride, valid padding, activation relu, output 500 |
| flatten               |                                           |
| Dropout               |   prob 0.2                                    |
| Fully connected       |                                               |
| Dropout               |   prob 0.4                                    |
| Fully connected       |                                               |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. And one lap reverse driving on track one.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase the size of the data and reduce overfitting. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also use the image from the side cameras. I adjust 0.2 angles for those images.

After the collection process, I had X number of data points. I then preprocessed this data by 32688.


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the learning curve. After 5 epochs the test accuracy stars to decrease, which means the model starts to overfit the data. I used an adam optimizer so that manually training the learning rate wasn't necessary.
