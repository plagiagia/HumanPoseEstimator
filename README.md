# Human Pose Estimator

## Introduction
This project is a solution for the live project on the [Manning publications website](https://www.manning.com/)
. The website provide some instructions and its up to each participant how to approach the solution. I find it quite interesting
and easy to follow, plus you have to do with something real and develop your skills to the next level.

## Chapter 1
The first chapter is about downloading the datasets and get familiar with them. The two datasets are the [SHVN](http://ufldl.stanford.edu/housenumbers/) 
and the [COCO](https://cocodataset.org/#keypoints-2017). Links are provided for the official websites.

The website provides a script to download the data from a S3 bucket on AWS cloud. After downloading the data 
I wrote a small script to visualise and understand the dataset. Running the pose_estimator_01.py you take the 
following results:

![img](static/image_with_box.png)

This image shoes that I successfully draw a box around the human and crop the image to the given coordinates

![img](static/image_with_keypoints.png)

Same image as before but after I found the box coordinates I plot the keypoints the represent all the visible
body parts of the human depicted on the picture

The jupyter notebook with all the code you can find here: [notebok]('01.HumanPoseEstimator.ipynb)

## Chapter 2