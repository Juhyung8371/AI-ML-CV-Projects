# Optical Character Recognition for Video Games

## Introduction
The goal of this project is to read the image of equipment in a video game called Maplestory and write down its description in an organized manner for better readability.

## Result
<img src='https://drive.google.com/uc?export=view&id=1bActswBbaAd_fJa-4MJU3fL0imxLb87n'>

## Challenges

1. Low text detection accuracy:

The biggest challenge came from the poor font of the equipment description. The little spacing between slim characters is the main problem. If slim characters are adjacent to each other, then they tend to cause confusion to the OCR.

Inaccurate OCR result examples:

'INT' into 'NT'
'Skill' into 'Ski'
'Attack' into 'Aitack'
'all' to 'ai'
'Power:' to 'Power.'

This is the inherent limit of the input, and I must work around it.

2. Slow text detection.

The entire process takes around 14 seconds. This is too slow. The most time (10+ seconds) is taken in the EasyOCR's text detection step. A faster model or solution is needed.

## Future work

EasyOCR is easy to work with since it's ready to use off the shelf. However, it is not too robust in terms of the types of font it can detect.

To solve this issue, I can make a new OCR model using YOLOv8. YOLO is a fast and accurate object detection model. I think I can train the model to recognize the unique traits of Maplestory's font. I can also train it to recognize the region of interest - where the important descriptions are located - so there is less post-processing to do to get the final result.

An example of training data might look like the image below. The large, beige-colored box is the region of interest where the important descriptions are. The rest of the smaller boxes are the important descriptions.

<img src='https://drive.google.com/uc?export=view&id=1Pysn4Xt-VHS1mZQmg0v02BJYuBVrKDq-'>

