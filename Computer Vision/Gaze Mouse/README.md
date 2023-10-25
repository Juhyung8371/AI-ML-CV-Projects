# Gaze-Controlled Mouse Control

## Abstract:
This study is a proof of concept for controlling a computer mouse pointer using a human gaze. This project involves face detection, eye feature extraction, blink detection, iris tracking, and computer mouse controls. The result showed that gaze can be an effective method to interact with a computer. It will be particularly useful for users who cannot physically control a mouse due to health complications.

## Introduction

A mouse is one of the most popular and cheap devices for humans to interact with computers. It allows the user to use their hands to control a mouse pointer accurately and reliably to communicate with a computer. However, a mouse is inaccessible for those without hand mobility due to injury and sickness. This project aims to provide an affordable and real-time solution to this issue by enabling mouse control using eye movements, such as blinking and gazing.

## Literature Review

Facial features, especially those of the eyes, are essential in human-to-human communication since they are rich in information. Because of that, many researchers explored the potential of combining facial features with the field of human-computer interaction.

hello [[1]](#1)



This paper \cite{reviews} offers a comprehensive analysis of existing gaze estimation systems. It identifies eye behavior types, gaze estimation algorithms, calibration methods, application platforms, performance evaluation metrics, and sources of errors in gaze estimation. First, eye behavior analysis offers movement type, execution rate, duration, significance, and possible applications. Second, it classifies gaze estimation algorithms into five types. 2D regression, 3D model, and cross ratio-based methods comprise of polynomial or geometrical analysis of the corneal reflection data from a near-infrared light source. Appearance and shape-based methods use visible light information from a camera, such as facial features, texture, and shape. Third, it explains that calibration is usually performed by asking the user to gaze at certain targets on the screen for some period. Speaking of calibration, since calibration is inconvenient for users, \cite{auto_calibration} attempts to automate it by comparing initial user data with the known saliency map to seamlessly calibrate the sensor via K-closest points and mixture model fitting. Fourth, it classifies application platforms as large, fixed screens (e.g. desktop and television), head-mounted devices (e.g. glasses, helmets, and goggles), automotive, and hand-held devices (e.g. phones and tablets). Fifth, it identifies the system performance evaluation metrics, including user parameters, peripheral devices, test environment, and accuracy metrics. Finally, it identifies major error factors, such as image resolution, display size, viewing angle and distance, head pose, platform movement, illumination level, and occlusion. These pieces of information were helpful in grasping the general trend of gaze estimation systems. In addition, \cite{evaluations} compares six pupil detection algorithms Starburst, Swirski, Pupil Labs, SET, ExCuSe, and ElSe. They are all based on traditional computer vision methods like contour detection, ellipse-fitting, Random Sample Consensus, Haar-like feature detector, image thresholding, and Canny edge detector.

The paper \cite{blink_con} proposes a method to use gaze tracking to maneuver the mouse pointer and blink detection to enable a mouse click. Its gaze-tracking algorithm employs the dlib face-tracking library to extract the eye regions from the picture taken by a webcam, divides the region into a 2×2 grid, and estimates the gaze by counting the number of black pixels from the user's iris in each section (the threshold depends on each user). Additionally, it detects a blink by determining the ratio between the length of the eye to the distance between the eyelids. The voluntary and involuntary blinks are differentiated by their duration.

This study \cite{fast_gaze} proposes a robust approach for detecting and tracking an iris quickly. Its approach uses a Haar-like feature-based method to detect a face, a Circular Hough transforms filter to estimate the iris location roughly, and radial ray casting similar to Starburst combined with the RANSAC algorithm to fit an ellipse around the iris. Once the iris is detected, it uses the Kalman filter to track it, so it can skip detecting the face again to reduce the computational load. It maps the vector between the eye center and the eye corner and uses a regression framework to determine the point of gaze. It extracts HoG features from the eye used to train the SVM classifier to determine the blink. Additionally, it is robust against an in-plane rotation of images by using an affine transform-based algorithm. 

The paper \cite{gaze_pointer} uses a pre-trained Haar cascade model to detect facial features from the picture taken by a webcam, uses Hough circle transformation to detect the iris, calculates the gaze projection based on the iris's relative displacement from the center of the eye, and use translate that displacement onto the movement of the mouse pointer. To increase its robustness, it calibrates the system by letting the user look at each corner of the computer screen to collect the iris displacement data. However, the author acknowledges the system's limitation in robustness against lighting changes and iris detection accuracy. It also does not discuss how smoothly the mouse moves.

Recently, deep learning methods have gained popularity in gaze estimation studies. Studies often employ convolutional neural network models with variances like using intermediate supervision \cite{deep_learning_gaze} and considering the more facial features like full face \cite{all_over_face, for_everyone}, head poses \cite{head_pose}, and blink \cite{rt_bene}. Deep learning approaches often yield high accuracy, often under 4° error in gaze estimation. However, the development of deep learning models requires a large dataset. Therefore, the proposed solution is based on traditional computer vision methods.

Gaze detection can also complement other human-computer interaction methods. A synthesis with speech recognition can be especially beneficial to the elderly, who may be reluctant to newer technological devices and facing health and communication problems from aging \cite{speech_gaze}.



## References

<a id="1">[1]</a> 
Neogi, Debosmit, Nataraj Das, and Suman Deb. ``BLINK-CON: A HANDS FREE MOUSE POINTER CONTROL WITH EYE GAZE TRACKING.'' 2021 IEEE Mysore Sub Section International Conference (MysuruCon). Piscataway: IEEE, 2021. 50–57. Web.

<a id="2">[2]</a> 
Ghani, Muhammad Usman et al. ``GazePointer: A Real Time Mouse Pointer Control Implementation Based on Eye Gaze Tracking.'' INMIC. IEEE, 2013. 154–159. Web.

\bibitem{deep_learning_gaze} Park, Seonwook, Adrian Spurr, and Otmar Hilliges. ``Deep Pictorial Gaze Estimation.'' arXiv.org. Ithaca: Cornell University Library, arXiv.org, 2018. Web.
\bibitem{rt_bene} K. Cortacero, T. Fischer and Y. Demiris, ``RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments,'' 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), Seoul, Korea (South), 2019, pp. 1159-1168, doi: 10.1109/ICCVW.2019.00147.
\bibitem{all_over_face} Zhang, Xucong et al.``It’s Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.'' arXiv.org. Ithaca: Cornell University Library, arXiv.org, 2023. Web.
\bibitem{auto_calibration} Alnajar, F et al. ``Auto-Calibrated Gaze Estimation Using Human Gaze Patterns.'' International journal of computer vision 124.2 (2017): 223–236. Web.
\bibitem{evaluations} Fuhl, Wolfgang et al. ``Pupil Detection for Head-Mounted Eye Tracking in the Wild: An Evaluation of the State of the Art.'' Machine vision and applications 27.8 (2016): 1275–1288. Web.
\bibitem{speech_gaze} Acartürk, Cengiz et al. “Elderly Speech-Gaze Interaction - State of the Art and Challenges for Interaction Design.” Interacción (2015).
\bibitem{fast_gaze} George, Anjith, and Aurobinda Routray. ``Fast and Accurate Algorithm for Eye Localisation for Gaze Tracking in Low-Resolution Images.'' IET computer vision 10.7 (2016): 660–669. Web.
\bibitem{for_everyone} Krafka, Kyle et al. ``Eye Tracking for Everyone.'' 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2016. 2176–2184. Web.
\bibitem{head_pose} Leblond-Menard, C, and S Achiche. ``Non-Intrusive Real Time Eye Tracking Using Facial Alignment for Assistive Technologies.'' IEEE transactions on neural systems and rehabilitation engineering PP (2023): 1–1. Web.

\bibitem{reviews} Kar, Anuradha, and Peter Corcoran. ``A Review and Analysis of Eye-Gaze Estimation Systems, Algorithms and Performance Evaluation Methods in Consumer Platforms.'' IEEE access 5 (2017): 16495–16519. Web.
\bibitem{ophthalmology} van den Bosch, Willem A, Ineke Leenders, and Paul Mulder. ``Topographic Anatomy of the Eyelids, and the Effects of Sex and Age.'' British journal of ophthalmology 83.3 (1999): 347–352. Web.

