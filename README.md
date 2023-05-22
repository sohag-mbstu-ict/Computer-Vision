
# Convolutional Neural Network (CNN)
 

A Convolutional Neural Network is a special class of neural networks that are built with the ability to extract unique features from image data. For instance, they are used in face detection and recognition because they can identify complex features in image data. 


### Working principle of Convolutional Neural Networks
Like other types of neural networks, CNNs consume numerical data.

Therefore, the images fed to these networks must be converted to a numerical representation. Since images are made up of pixels, they are converted into a numerical form that is passed to the CNN.


### Convolution
The purpose of the convolution is to extract the features of the object on the image locally. It means the network will learn specific patterns within the picture and will be able to recognize it everywhere in the picture.

### feature map or convolved feature or activation map

Convolution is an element-wise multiplication. The concept is easy to understand. The computer will scan a part of the image, usually with a dimension of 3×3 and multiplies it to a filter. The output of the element-wise multiplication is called a feature map, convolved feature, or activation map. This step is repeated until all the image is scanned. Note that, after the convolution, the size of the image is reduced


### Feature detector or kernel or filter
Applying a feature detector is what leads to a feature map convolved feature, or activation map. The feature detector is also known by other names such as kernel or filter.


### Depth
It defines the number of filters to apply during the convolution. Depth of 1 meaning only one filter is used. In most of the case, there is more than one filter. 


### Stride
It defines the number of “pixel’s jump” between two slices. If the stride is equal to 1, the windows will move with a pixel’s spread of one. If the stride is equal to two, the windows will jump by 2 pixels. If you increase the stride, you will have smaller feature maps.

### Ppadding
A padding is an operation of adding a corresponding number of rows and column on each side of the input features maps. In this case, the output has the same dimension as the input.

When building the CNN, you will have the option to define the type of padding you want or no padding at all. The common options here are valid or same. Valid means no padding will be applied while same means that padding will be applied so that the size of the feature map is the same as the size of the input image.

### Activation functions
At the end of the convolution operation, the output is subject to an activation function to allow non-linearity. The usual activation function for convnet is the Relu. All the pixel with a negative value will be replaced by zero.

### Pooling Operation
This step is easy to understand. The purpose of the pooling is to reduce the dimensionality of the input image. The steps are done to reduce the computational complexity of the operation. By diminishing the dimensionality, the network has lower weights to compute, so it prevents overfitting.

A common technique is max-pooling. The size of the pooling filter is usually a 2 by 2 matrix. In max-pooling, the 2 by 2 filter slides over the feature map and picks the largest value in a given box. This operation results in a pooled feature map.

### Dropout Regularization
Applying Dropout Regularization is a common practice in CNNs. This involves randomly dropping some nodes in layers so that they are not updated during back-propagation. This prevents overfitting.

### Flattening
Flattening involves transforming the pooled feature map into a single column that is passed to the fully connected layer. This is a common practice during the transition from convolutional layers to fully connected layers.

### Fully connected layers
The flattened feature map is then passed to a fully connected layer. There might be several fully connected layers depending on the problem and the network. The last fully connected layer is responsible for outputting the prediction. 

An activation function is used in the final layer depending on the type of problem. A sigmoid activation is used for binary classification, while a softmax activation function is used for multi-class image classification.
