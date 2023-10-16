# Spot the difference

I will make a program that finds the difference between two images. I tested three different methods:

1. HSV filter
2. Pillow ImageChop
3. Structural Similarity (SSIM)

The real-life application of spotting the difference includes:

1. Image quality check after processing (compression, resizing, etc.)
2. Security system (spot moving object or person)
3. Inventory check (spot the empty shelf)

___

### HSV filter method

Determine the difference by comparing the HSV color value. It gives a decent result but is weak against spots with little color difference. 

___

### ImageChop

Use Pillow's ImageChop method. It is a similar method to my HSV difference method, but better. 

___

### Structural similarity

This paper, [Image Quality Assessment: From Error Visibility to
Structural Similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf) proposes the Structural Similarity (SSIM) Index as a metric to measure image similarity using luminance, contrast, and structural information. This is an improvement over simpler methods like mean squared error comparison. 

I used scikit-image's implementation of SSIM. It performs very well in different styles of images (real-life, clip art, etc.) and can detect small and faint differences as well. 

___

### Result - Combine all three

Since structural similarity performed the best, I will use that as a base and reinforce that with the result of HSV filtered image and ImageChop image.




