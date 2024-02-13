# Gaze-Controlled Mouse Control

## Abstract:
This study is a proof of concept for controlling a computer mouse pointer using a human gaze. This project involves face detection, eye feature extraction, blink detection, iris tracking, and computer mouse controls. The result showed that gaze can be an effective method to interact with a computer. It will be particularly useful for users who cannot physically control a mouse due to health complications.

## Introduction

A mouse is one of the most popular and cheap devices for humans to interact with computers. It allows the user to use their hands to control a mouse pointer accurately and reliably to communicate with a computer. However, a mouse is inaccessible for those without hand mobility due to injury and sickness. This project aims to provide an affordable and real-time solution to this issue by enabling mouse control using eye movements, such as blinking and gazing.

## Literature Review

Facial features, especially those of the eyes, are essential in human-to-human communication since they are rich in information. Because of that, many researchers explored the potential of combining facial features with the field of human-computer interaction.

This paper [[12]](#12) offers a comprehensive analysis of existing gaze estimation systems. It identifies eye behavior types, gaze estimation algorithms, calibration methods, application platforms, performance evaluation metrics, and sources of errors in gaze estimation. First, eye behavior analysis offers movement type, execution rate, duration, significance, and possible applications. Second, it classifies gaze estimation algorithms into five types. 2D regression, 3D model, and cross ratio-based methods comprise of polynomial or geometrical analysis of the corneal reflection data from a near-infrared light source. Appearance and shape-based methods use visible light information from a camera, such as facial features, texture, and shape. Third, it explains that calibration is usually performed by asking the user to gaze at certain targets on the screen for some period. Speaking of calibration, since calibration is inconvenient for users, [[6]](#6) attempts to automate it by comparing initial user data with the known saliency map to seamlessly calibrate the sensor via K-closest points and mixture model fitting. Fourth, it classifies application platforms as large, fixed screens (e.g. desktop and television), head-mounted devices (e.g. glasses, helmets, and goggles), automotive, and hand-held devices (e.g. phones and tablets). Fifth, it identifies the system performance evaluation metrics, including user parameters, peripheral devices, test environment, and accuracy metrics. Finally, it identifies major error factors, such as image resolution, display size, viewing angle and distance, head pose, platform movement, illumination level, and occlusion. These pieces of information were helpful in grasping the general trend of gaze estimation systems. In addition, [[7]](#7) compares six pupil detection algorithms: Starburst, Swirski, Pupil Labs, SET, ExCuSe, and ElSe. They are all based on traditional computer vision methods like contour detection, ellipse-fitting, Random Sample Consensus, Haar-like feature detector, image thresholding, and Canny edge detector.

The paper [[1]](#1) proposes a method to use gaze tracking to maneuver the mouse pointer and blink detection to enable a mouse click. Its gaze-tracking algorithm employs the dlib face-tracking library to extract the eye regions from the picture taken by a webcam, divides the region into a 2×2 grid, and estimates the gaze by counting the number of black pixels from the user's iris in each section (the threshold depends on each user). Additionally, it detects a blink by determining the ratio between the length of the eye to the distance between the eyelids. The voluntary and involuntary blinks are differentiated by their duration.

This study [[9]](#9) proposes a robust approach for detecting and tracking an iris quickly. Its approach uses a Haar-like feature-based method to detect a face, a Circular Hough transforms filter to estimate the iris location roughly, and radial ray casting similar to Starburst combined with the RANSAC algorithm to fit an ellipse around the iris. Once the iris is detected, it uses the Kalman filter to track it so it can skip detecting the face again to reduce the computational load. It maps the vector between the eye center and the eye corner and uses a regression framework to determine the point of gaze. It extracts HoG features from the eye used to train the SVM classifier to determine the blink. Additionally, it is robust against an in-plane rotation of images by using an affine transform-based algorithm. 

The paper [[2]](#2) uses a pre-trained Haar cascade model to detect facial features from the picture taken by a webcam, uses Hough circle transformation to detect the iris, calculates the gaze projection based on the iris's relative displacement from the center of the eye, and use translate that displacement onto the movement of the mouse pointer. To increase its robustness, it calibrates the system by letting the user look at each corner of the computer screen to collect the iris displacement data. However, the author acknowledges the system's limitation in robustness against lighting changes and iris detection accuracy. It also does not discuss how smoothly the mouse moves.

Recently, deep learning methods have gained popularity in gaze estimation studies. Studies often employ convolutional neural network models with variances like using intermediate supervision [[3]](#3) and considering more facial features like full face [[5]](#5) [[10]](#10), head poses [[11]](#11), and blink [[4]](#4). Deep learning approaches often yield high accuracy, often under 4° error in gaze estimation. However, the development of deep learning models requires a large dataset. Therefore, the proposed solution is based on traditional computer vision methods.

Gaze detection can also complement other human-computer interaction methods. A synthesis with speech recognition can be especially beneficial to the elderly, who may be reluctant to newer technological devices and face health and communication problems from aging [[8]](#8)

## Proposed Solution

### A. Summary

The proposed solution includes gaze detection for controlling a mouse pointer and blink detection for clicking. The input is the image of the user looking at the monitor, and the output is the mouse movement, clicking, and double-clicking. 

### B. Experimental Setup

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/setup.png' width=250>

The figure above demonstrates the experimental setup. The user's face is approximately 60 cm away from the monitor and the webcam. The camera used for this experiment is HP TrueVision HD Camera with 1280×720 resolution, 30 frames per second capture rate, and auto-focus feature.

There are some constraints for the setup to ensure accurate results: 

1. The camera must capture the user's full face. The ideal camera location is right above the computer monitor, at around the eye level of the user.
2. The user and the camera must face each other - the face's roll, yaw, and pitch changes are not allowed. The user needs to center his face in the template shown below:

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/tilt.gif' width=300>

### C. Face Detection and Eye Region Extraction

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/mediapipe_landmarks.jpg' width=300 align='right'>

The proposed solution employs MediaPipe's face landmark detection model to detect the face and extract facial features. It detects faces in the image and adds landmarks to them in a 3D space, as shown below. These landmarks are used to extract the eye features for blink detection and gaze detection (see [the list of landmarks](https://github.com/tensorflow/tfjs-models/blob/ad17ade67add3e84fee0895c938ea4e1cd4d50e4/face-landmarks-detection/src/constants.ts)). Since the input image is taken 60 cm away from the user's face using an affordable webcam, which limits the amount and the quality of facial information, the solution required unique approaches described in the later sections.   


As a side note, I initially attempted them using the dlib face recognition model. However, it was too weak for my application. For example, it only has 68 facial landmarks, does not detect iris, is weak against change in illumination, has trouble recognizing faces with accessories on, etc. Also, since it did not have iris landmarks, I needed to detect the iris using computer vision techniques such as thresholding and segmentation, which introduced extra layers of complexity and computational load.

### D. Blink Detection

The blink detection algorithm is an appearance-based method similar to the method from [[1]](#1). These are the steps for the blink-controlled mouse clicker algorithm:

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/fissure_size.png' width=300>

1. Take the eye landmarks obtained from the MediaPipe face mesh.
2. Take the ratio between the vertical and horizontal eyelid fissure size. The figure above shows that it is a ratio of CD/AB. I define this ratio as a blink ratio.
3. Collect the blink ratios over time and identify the pattern that rises from a voluntary blink. Since more closed eyelids yield a higher blink ratio, the blink pattern should be low-high-low. In particular, I determined that 6 is an adequate threshold after hyperparameter tuning and data from [[13]](#13) (Horizontal fissure size averaged 25.8 mm (SD 2.4 mm) in men, and 25.1 mm (SD 2.1 mm) in women.)

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/blink_ratio.png' width=400>

The figure above plots the gaze ratio over time. We can see that blinks are demonstrated as spikes in the plot. Therefore, I detected those single and double spikes in the plot and connected them to single-click and double-click features accordingly.

### E. Gaze Detection

Gaze can move vertically and horizontally. The core idea behind my gaze detection method is to find the threshold points that will stay similar regardless of face size or facial movement. I determined that the nose landmarks are the most stable landmarks, so I referenced many threshold values based on the nose tip landmark.
 
#### Horizontal Gaze

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/hor_thresh.jpg' width=350>

The horizontal gaze is determined by checking whether the center point between pupils lies between the horizontal thresholds from the nose landmarks (44 and 274). For example, like the figure above, if the user looks left, his pupil center will shift past the left threshold. 

#### Vertical Gaze

Vertical gaze detection is a bit less intuitive than horizontal gaze detection. I cannot use the pupil's relative position against the eye to determine its vertical position. It's due to the limitation of MediaPipe face mesh: the iris landmarks always stay within the eye fissure landmarks. In other words, if the person looks up and the iris moves up, all the eye landmarks will follow the iris instead of just the iris moving up. See the figures below to check this in action.

So, instead of using the pupil's relative position against the eye fissure as the measure to determine its vertical motion, I measured its distance to the tip of the nose. To be more accurate, I measure the following ratio:

> ratio = Nose_to_pupil / NoseTop_to_NoseBottom

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/up.gif' height=350> <img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/ver_thresh.jpg' height=350>

NoseTop_to_NoseBottom will stay relatively the same all the time, whereas Nose_to_pupil will change depending on the vertical position of the iris.

## Result

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/click.gif' width=700>

I moved the mouse using my gaze and opened the file using double-blink. In terms of precision, I can click the Windows icon I want to click, but I can't reliably click between small characters (12pt font) in a text editor to put my cursor between characters. 

### More Evaluation

I tested the solution with test images, and the following are the results:

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/result1.jpg' width=300> <img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/result5.jpg' width=300>

Some results worked as intended. Good.

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/result2.jpg' width=300> <img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/result4.jpg' width=300>

Some results say the user is looking away from the camera when facing it correctly. This is caused by the hard-coded rotation thresholds I set based on my facial features. 

<img src='https://raw.githubusercontent.com/Juhyung8371/AI-ML-CV-Projects/main/Computer%20Vision/Gaze%20Mouse/images/result3.jpg' width=300>

This result says the user is looking down when he is looking right. This is also caused by the hard-coded directional thresholds I set based on my facial features. 

## Discussion

This solution can improve in robustness. Many hard-coded constants, such as threshold values, are fine-tuned for my face. To solve this problem, I can:

1. Introduce a calibration session to collect the user's unique facial features. For instance, I can ask the user to look up, down, left, and right to collect the appropriate thresholds. I can also occasionally do some non-intrusive calibration during runtime to reinforce the calibration.
   
2. Create a machine learning model that can determine the gaze direction. As the Literature Review section discusses, machine learning can be a robust and reliable solution. However, it will require a lot of training data.
   
3. I can enhance the detection result with computer vision techniques like adaptive thresholding, noise filtering, and blob detection. Those traditional computer vision approaches were mostly removed when I changed the face detection model from the dlib to MediaPipe, but I can bring them back to reinforce the eye detection result from the ML model.

There are some usability issues, too:

1. The mouse control could be more precise. As I said, it needs to be more precise to click between small characters, which means it cannot assist users with text editing tasks like writing an email. I can make the mouse movement more precise by making the cursor slower. However, that will reduce the usability since the user will have to look away from the screen longer.
   
2. Head movement is quite restricted. Although a certain level of restriction is required to ensure the quality of face feature data, it's hard to fix the face in one spot for too long. I can adaptively update the threshold values based on the head movement to accommodate more various head positions. For example, if the user is slightly looking to the right, I can slightly shift the thresholds to the left.
 
Despite its shortcomings, my solution demonstrates that gaze is a feasible human-computer interaction method. It's reliable and comfortable for tasks that don't require very precise movements.

## References

<a id="1">[1]</a> 
Neogi, Debosmit, Nataraj Das, and Suman Deb. "BLINK-CON: A HANDS FREE MOUSE POINTER CONTROL WITH EYE GAZE TRACKING." 2021 IEEE Mysore Sub Section International Conference (MysuruCon). Piscataway: IEEE, 2021. 50–57. Web.

<a id="2">[2]</a> 
Ghani, Muhammad Usman et al. "GazePointer: A Real Time Mouse Pointer Control Implementation Based on Eye Gaze Tracking." INMIC. IEEE, 2013. 154–159. Web.

<a id= "3">[3]</a> Park, Seonwook, Adrian Spurr, and Otmar Hilliges. "Deep Pictorial Gaze Estimation." arXiv.org. Ithaca: Cornell University Library, arXiv.org, 2018. Web.

<a id= "4">[4]</a> K. Cortacero, T. Fischer and Y. Demiris, "RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments," 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), Seoul, Korea (South), 2019, pp. 1159-1168, doi: 10.1109/ICCVW.2019.00147.

<a id= "5">[5]</a> Zhang, Xucong et al. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." arXiv.org. Ithaca: Cornell University Library, arXiv.org, 2023. Web.

<a id="6">[6]</a> Alnajar, F et al. ``Auto-Calibrated Gaze Estimation Using Human Gaze Patterns.'' International journal of computer vision 124.2 (2017): 223–236. Web.

<a id= "7">[7]</a> Fuhl, Wolfgang et al. "Pupil Detection for Head-Mounted Eye Tracking in the Wild: An Evaluation of the State of the Art." Machine vision and applications 27.8 (2016): 1275–1288. Web.

<a id= "8">[8]</a> Acartürk, Cengiz et al. "Elderly Speech-Gaze Interaction - State of the Art and Challenges for Interaction Design." Interacción (2015).

<a id="9">[9]</a> George, Anjith, and Aurobinda Routray. "Fast and Accurate Algorithm for Eye Localisation for Gaze Tracking in Low-Resolution Images." IET computer vision 10.7 (2016): 660–669. Web.

<a id="10">[10]</a> Krafka, Kyle et al. ``Eye Tracking for Everyone.'' 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2016. 2176–2184. Web.

<a id="11">[11]</a> Leblond-Menard, C, and S Achiche. "Non-Intrusive Real Time Eye Tracking Using Facial Alignment for Assistive Technologies." IEEE transactions on neural systems and rehabilitation engineering PP (2023): 1–1. Web.

<a id="12">[12]</a> Kar, Anuradha, and Peter Corcoran. "A Review and Analysis of Eye-Gaze Estimation Systems, Algorithms and Performance Evaluation Methods in Consumer Platforms." IEEE access 5 (2017): 16495–16519. Web.

<a id="13">[13]</a> van den Bosch, Willem A, Ineke Leenders, and Paul Mulder. "Topographic Anatomy of the Eyelids, and the Effects of Sex and Age." British journal of ophthalmology 83.3 (1999): 347–352. Web.

