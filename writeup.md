# **Behavioral Cloning** 

## Writeup


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* output.mp4 which shows a video if the autonomously driven car simulator using the above saved model

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Instead of starting out from scratch I reaserched on the internet for similar papers where end to end autonomous driving using CNN has been employed. 

During this search I found the NVIDIA paper (also suggested by Udacity) and another solution my comma.ai. Because of the simplicity of the model I chose to base my model on the comma.ai model.

1. My model starts with a Lambda layer to normalize the input on the fly.
2. As there is lot of irrelevant information in the field of view of the car, I added a cropping layer to remove the top 75 pixel and botton 25 pixel width of the image.

    The following code snipped describes the model. 

```sh
    # Model
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((75,25), (0,0))))
    model.add(Conv2D(16, kernel_size=(8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(.5))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
```

The model includes ELU layers to introduce nonlinearity. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting with a keep probability of 0.5. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

I generated data of center lane driving for 2 laps and 1 lap of intermittent recording where data was captured only for recovering from the side of the lanes. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

Along with the center image I used the left and right camera images where a correction factor of 0.2 was applied to the steering angle. 

I also normalized the sample images by removed close to 50% of the images where the steering angle was less than 0.2 both towards left or right. This made sure that we had similar number of samples driving straight lines and curved. 

I did not augment the data further as the results from this data was able to drive the car on the center of the track. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by training and validation loss values. 


## Next Steps:

* Use the generator function to reduce the memory usage. (I implemented this method and found that the models generated were less accurate). I will further investigate this and what is causing the poor accuracy when generator is used. 
* Collect data from the track 2 and train the model to work on both the tracks
* Implement other Model architectures and understand what hyperparameters are effecting the accuracy of the model and find out the reasons why.