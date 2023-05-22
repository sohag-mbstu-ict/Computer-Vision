
# Convolutional Neural Network (CNN)
 

A Convolutional Neural Network is a special class of neural networks that are built with the ability to extract unique features from image data. For instance, they are used in face detection and recognition because they can identify complex features in image data. 


# Working principle of Convolutional Neural Networks
Like other types of neural networks, CNNs consume numerical data.

Therefore, the images fed to these networks must be converted to a numerical representation. Since images are made up of pixels, they are converted into a numerical form that is passed to the CNN.


# Convolution
The purpose of the convolution is to extract the features of the object on the image locally. It means the network will learn specific patterns within the picture and will be able to recognize it everywhere in the picture.

# feature map or convolved feature or activation map

Convolution is an element-wise multiplication. The concept is easy to understand. The computer will scan a part of the image, usually with a dimension of 3×3 and multiplies it to a filter. The output of the element-wise multiplication is called a feature map, convolved feature, or activation map. This step is repeated until all the image is scanned. Note that, after the convolution, the size of the image is reduced


# Feature detector or kernel or filter
Applying a feature detector is what leads to a feature map convolved feature, or activation map. The feature detector is also known by other names such as kernel or filter.


# Depth
It defines the number of filters to apply during the convolution. Depth of 1 meaning only one filter is used. In most of the case, there is more than one filter. 


# Stride
It defines the number of “pixel’s jump” between two slices. If the stride is equal to 1, the windows will move with a pixel’s spread of one. If the stride is equal to two, the windows will jump by 2 pixels. If you increase the stride, you will have smaller feature maps.

# Zero-padding
A padding is an operation of adding a corresponding number of rows and column on each side of the input features maps. In this case, the output has the same dimension as the input.
