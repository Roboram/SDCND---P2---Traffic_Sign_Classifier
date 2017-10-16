#**Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Writeup-Images/Visualisation.png "Visualization"
[image2]: ./Writeup-Images/Histogram.png "Histogram"
[image3]: ./Writeup-Images/RGB-GRY.png "RGB-GRY"
[image4]: ./Writeup-Images/Normalized.png "Normalized"
[image5]: ./Writeup-Images/Probability.png "Top Probability"
[image6]: ./Writeup-Images/New-Images.png "New Images"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

To visualise the number of images under each class I used a bar chart. It shows how the data is spread across the 43 labels.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because colour doesnt add much necessary information to a traffic sign. Also it reduces the dimension of the image thus using less memory and reducing the training time.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data to standardize the inputs to the network. It makes learning more stable by reducing variability and the network will generalize better to novel data.


Here is an example of an original image and an normalized image:

![alt text][image4]

 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 RGB image   				| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			| Activation					|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			|
| Convolution 3x3	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU			| Activation					|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			|
| Fully connected	| Output 120        				|
| RELU			| Activation					|
| Dropout		| keep probability = 0.5			|
| Fully connected	| Output 84       				|
| RELU			| Activation					|
| Dropout		| keep probability = 0.5			|
| Fully connected	| Output 43        				|
| Softmax		| Outputs 'logits'.        			|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer. The final settings used were:

    1. batch size: 128
    2. epochs: 50
    3. learning rate: 0.0009
    4. mu: 0
    5. sigma: 0.1
    6. dropout keep probability: 0.5


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8 %
* validation set accuracy of 95.8 % 
* test set accuracy of 94.7 %

I used the LeNet architecture thought in the lessons. I used 3 convolution layers and 3 fully connected layers with dropout probability of 0.5. The training accuracy was a little bit high than validation accuracy. This little overfitting might have caused because of the highly fluctuating number of images for each label in the training data set. I raised the number of epochs and reduced the learning rate parameter to improve the accuracy of prediction.

The final model is robust as the the test set accuracy is near to 95 % 


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The second image will be difficult to classify beacuase the number of images in the training set is low followed by the fourth image followed by the third image. The other two images are easy to predict compared to other images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			       	     |     Prediction	        		| 
|:----------------------------------:|:----------------------------------------:| 
| Speed Limit 30      		     | Speed Limit 30    			| 
| Keep Right     		     | Keep Right 				|
| Right of way at next intersection  | Right of way at next intersection	|
| Turn Left Ahed	      	     | Turn Left Ahed				|
| Priority Road			     | Priority Road		    		|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.7 %.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.


| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         		| Speed Limit 30   				| 
| 1.0     		| Keep Right 					|
| 1.0			| Right of way at next intersection		|
| 1.0	      		| Turn Left Ahed				|
| 1.0			| Priority Road      				|


For all the 5 new images downloaded from the web, the predictions of the labels were accurate to 100%. This might be because the images were clear and bright. I guess the model might fail if the images have strange contast or brightness values. Hence this proves my robustness of my model on the new data.




