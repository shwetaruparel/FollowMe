[//]: # (Image References)
[image_0]: ./images/cnn.png
[image_1]: ./images/conv_1_1.png
[image_2]: ./images/ConvNet.png
[image_3]: ./images/FCN.png
[image_4]: ./images/architecture-cnn.png
[image_5]: ./images/encdec.png
[image_6]: ./images/bilin.png
[image_7]: ./images/herotarget.png
[image_8]: ./images/lossplot.png
[image_9]: ./images/learningrate.png


# FollowMe
Drone tracking and following a single hero target 

## Deep Learning Project ##

This project will help build your own segmentation network, train it, valide it and deploy it . The trained deep neural network will be used to identify a target person from images produced by a quadcoptor simulator. 

The Original Project Repository is here--> https://github.com/udacity/RoboND-DeepLearning-Project.git

The model is built within Tensorflow and Keras, and was trained using UDACITY GPU Workspace

### Network Architecture

#### Deep neural networks contain multiple non-linear hidden layers and this makes them very expressive models that can learn very           complicated relationships between their inputs and outputs.
  
*Convolutional networks are powerful visual models that yield hierarchies of features and useful for image classification, object detection, and recognition tasks.CNNs are implemented as a series of interconnected layers.The layers are made up of repeated blocks of convolutional, ReLU (rectified linear units), and pooling layers. The convolutional layers convolve their input with a set of filters. The filters were automatically learned during network training. The ReLU layer adds nonlinearity to the network, which enables the network to approximate the nonlinear mapping between image pixels and the semantic content of an image. The pooling layers downsample their inputs and help consolidate local image features.*

*1)This helps CNN to learn classifying local patterns, like shapes and objects in an image.*

*2)It's common to have more than one filter.Different filters pick up different qualities of a patch. The amount of filters in a            convolutional layer is called the filter depth.If we have a depth of k, we connect each patch of pixels to k neurons in the next        layer. This gives us the height of k in the next layer.Multiple neurons can be useful because a patch can have multiple interesting      characteristics that we want to capture.*
  
![stride and filter][image_0]

*3)Multiple convultions layers are then finally connected to Fully Connected Layers followed by softmax activation function.*
  
*4)CNN isn't "programmed" to look for certain characteristics. Rather, it learns on its own which characteristics to notice.*
 
*5)Fully connected layer — Fully connected layers connect every neuron in one layer to every neuron in another layer. It is in principle the same as the traditional multi-layer perceptron neural network.*

*Finally, after several convolutional and max pooling layers, the high-level reasoning in the neural network is done via fully connected layers. Neurons in a fully connected layer have connections to all activations in the previous layer, as seen in regular neural networks. Their activations can hence be computed with a matrix multiplication followed by a bias offset.*

*Advantages: A fully connected layer learns features from all the combinations of the features of the previous layer, where a convolutional layer relies on local spatial coherence with a small receptive field.*

*Disadvantages: Fully connected layers are incredibly computationally expensive. That’s why we use them only to combine the upper layer features.Also fully connected layer don't preserve the spatial information.*

![Fully Connected Layer][image_4]

***The following image shows ConvNet Architecture*** 

![ConvNet Architecture][image_2]

   
***This is a great architecture of classifying task.***
   
***But,the question is - what if the image classification is needed in addition to location of the object in the image?***

### A Fully Convolutional Network is able to both identify the object and identify where it is

FCN is a powerful type of Neural Network, capable of carrying out complex computer vision tasks such as identifying objects in  an image. However, unlike a simple classifier, it is capable of showing where in the image the object of interest is located.This is the architecture we have used in the task follow me , where a human target is first identified and then followed as per the location of human target in the scene.

![Human Target][image_7]

The FCN is built to be able to segment objects within the video stream. This means that each pixel in the image needs to be              labeled. Fully convolutional networks are capable of this via a process called semantic segmentation. The model is built such            that the output image is the same size at the original input image.Semantic segmentation allows FCNs to preserve spatial                information throughout the network.
       
***Semantic Segmentation***
    
Semantic Segmentation is the cutting edge of perception in Robotics for full scene understanding.Semantic Segmentation is the task      of assigning mesaning to parts of an object. This can be done at pixel level, where each pixel can be assigned a target class.  

***Fully Convolutional Networks***

FCNs take advantage of three special techniquess

1) Replace Fully Connected Layer by 1 X 1 Convolutional Layer.

2) Upsampling through the use of bilinear/Transpose convolutional Layers.

3) Skip Connections to allow the network to use information from multiple resolution scales to make more precise segmentation decision.

***An FCN is usually comprised of two parts Encoder and Decoder.***

![Encoder and Decoder][image_5]

***Encoder***

 The encoder section is comprised of series of convolutional layers to extract features.Here we use Separable Convolution Layer

*Separable convolution layers are a convolution technique for increasing model performance by reducing the number of parameters in each convolution. Separable convolutions, also known as depthwise separable convolutions, comprise of a convolution performed over each channel of an input layer and followed by a 1x1 convolution that takes the output channels from the previous step and then combines them into an output layer. This technique allows for the efficient use of parameters. it is highly computationally efficient whilst also being extremely  accurate.

*Suppose we have an input shape of 32x32x3. With the desired number of 9 output channels and filters (kernels) of shape 3x3x3. In the regular convolutions, the 3 input channels get traversed by the 9 kernels.* ***That's a total of 243(9*3*3*3) parameters.***

*In case of the* ***separable convolutions***, *the 3 input channels get traversed with 1 kernel each. That gives us 27 parameters (3*3*3) and 3 feature maps. In the next step, these 3 feature maps get traversed by 9 1x1 convolutions each. That results in a total of 27 (9*3) parameters. That's a total of 54 (27 + 27) parameters! Way less than the 243 parameters we got above. And as the size of the layers or channels increases, the difference will be more noticeable.*

*The reduction in the parameters make separable convolutions quite efficient with improved runtime performance and are also, as a result, useful for mobile applications. They also have the added benefit of reducing overfitting to an extent, because of the fewer parameters.*
*The batch normalization layer has a number of advantages. It makes the network train more quickly and effectively and makes it easier to find good hyperparameters. It normalises the inputs of each layer so that they have a mean output activation of zero and standard deviation of one*
***Few advantages of using Batch Normalisation are:***

**Networks train faster –** Each training iteration will actually be slower because of the extra calculations during the forward pass. However, it should converge much more quickly, so training should be faster overall.

**Allows higher learning rates –** Gradient descent usually requires small learning rates for the network to converge. And as networks get deeper, their gradients get smaller during back propagation so they require even more iterations. Using batch normalization allows us to use much higher learning rates, which further increases the speed at which networks train.

**Simplifies the creation of deeper networks –** Because of the above reasons, it is easier to build and faster to train deeper neural networks when using batch normalization.

**Provides a bit of regularization –** Batch normalization adds a little noise to your network. In some cases, such as in Inception modules, batch normalization has been shown to work as well as dropout.

***1x1 Convolution Layer***

In TensorFlow, the output shape of a convolutional layer is a 4D tensor. However, when we wish to feed the output of a convolutional layer into a fully connected layer, we flatten it into a 2D tensor. This results in the loss of spatial information, because no information about the location of the pixels is preserved.We can avoid that by using 1x1 convolutions.

*The 1x1 convolution layer is a regular convolution, with a kernel and stride of 1. Using a 1x1 convolution layer allows the network to  be able to retain spatial information from the encoder. The 1x1 convolution layers allows the data to be both flattened for        classification while retaining spatial information. 1x1 convolution helped in reducing the dimensionality of the layer. A fully-connected layer of the same size would result in the same number of features. However, replacement of fully-connected layers with convolutional layers presents an added advantage that during inference (testing your model), you can feed images of any size into your trained network.*

![1X1 Convolution][image_1]

***The encoder layers allows model to gain a better understanding of the characeristics in the image, building a depth of understanding    with respect to specific features and thus the 'semantics' of the segmentation. The first layer might discern colours and brightness, the next might discern aspects of the shape of the object, so for a human body, arms and legs and heads might begin to become          successfully segmented. Each successive layers builds greater depth of semantics necessary for the segmentation. However, the deeper     the network, the more computationally intensive it becomes to train.***


***Decoder***

  The decoder section of the model can either be composed of transposed convolution layers or bilinear upsampling layers.

  The transposed convolution layers the inverse of regular convolution layers, multiplying each pixel of the input with the kernel.

  Bilinear upsampling is similar to 'Max Pooling' and uses the weighted average of the four nearest known pixels from the given pixel,     estimating the new pixel intensity value. Although bilinear upsampling loses some details it is much more computationally efficient     than transposed convolutional layers.

  ![Bilinear Sampling ][image_6]
  
***The bilinear upsampling method does not contribute as a learnable layer like the transposed convolutions in the architecture and is prone to lose some finer details, but it helps speed up performance.**

The decoder block mimics the use of skip connections by having the larger decoder block input layer act as the skip connection. It       calculates the separable convolution layer of the concatenated bilinear upsample of the smaller input layer with the larger input       layer.

***Skip Connections***

  Skip connections allow the network to retain information from prior layers that were lost in subsequent convolution layers. Skip         layers use the output of one layer as the input to another layer. By using information from multiple image sizes, the model retains     more information through the layers and is therefore able to make more precise segmentation decisions.

***Each decoder layer is able to reconstruct a little bit more spatial resolution from the layer before it. The final decoder layer will   output a layer the same size as the original model input image, which will be used for guiding the quad drone.***


### The FCN model used for the project contains a four encoder block layers,  1x1 convolution layers, and four decoder block layers.
   
    conv_in = conv2d_batchnorm(input_layer=inputs, filters=16, kernel_size=1, strides=1)
    enc_1 = encoder_block(input_layer=conv_in,  filters=32,  strides=2)
    # img_w/2 x img_h/2 x 32 => img_w/4 x img_h/4 x 64
    enc_2 = encoder_block(input_layer=enc_1,   filters=64,  strides=2)
    # img_w/4 x img_h/4 x 64 => img_w/8 x img_h/8 x 128
    enc_3_1 = encoder_block(input_layer=enc_2, filters=128, strides=2)
    # img_w/8 x img_h/8 x 128 => img_w/8 x img_h/8 x 128
    enc_3_2 = conv2d_batchnorm(input_layer=enc_3_1, filters=128, kernel_size=1, strides=1)
    # img_w/8 x img_h/8 x 128 => img_w/16 x img_h/16 x 256
    enc_4 = encoder_block(input_layer=enc_3_2, filters=256, strides=2)
    
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.

    conv_1x1 = conv2d_batchnorm(input_layer=enc_4,    filters=256, kernel_size=1, strides=1)
    # img_w/16 x img_h/16 x 256 => img_w/16 x img_h/16 x 128
    conv_1x1 = conv2d_batchnorm(input_layer=conv_1x1, filters=128, kernel_size=1, strides=1)        
    dec_1 = decoder_block(small_ip_layer=conv_1x1, large_ip_layer=enc_3_1,  filters=128)
    # img_w/8 x img_h/8 x 128 => img_w/4 x img_h/4 x 64
    dec_2 = decoder_block(small_ip_layer=dec_1,    large_ip_layer=enc_2,  filters=64)
    # img_w/4 x img_h/4 x 64 => img_w/2 x img_h/2 x 32
    dec_3 = decoder_block(small_ip_layer=dec_2,    large_ip_layer=enc_1,  filters=32)
    # img_w/2 x img_h/2 x 32 => img_w x img_h x num_classes
    x = decoder_block(small_ip_layer=dec_3,        large_ip_layer=inputs, filters=num_classes)   
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

#### Hyperparameters

Hyperparameters were found mostly via manual tuning and inspection. Starting with a learning rate of .01 to reaching 0.001 that gives the ***final grade score of 43% and final IOU of 57%***. There are 4131 images in the training dataset.Taking into consideration that no memory issues come around and no overfitting happens batch_size of 100 is chosen and accordingly selected steps_per_epoch to be 50 approximated by dividing number of images by batch_size.To maximize the speed 4 workers are chosen and it worked.validation_step was set around 4 times smaller than steps_for_epoch parameter.

![Loss Plot][image_8]

***Learning Rate was the most challenging parameter, I tried higher rates like 0.01 to lower rates of.009 to 0.001 and best parameter to be able to give better grade score was 0.001.***

*Learning rate is a hyper-parameter that controls how much we adjusting the weights of our network with respect to the loss gradient.While this might be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it could also mean that we’ll be taking a long time to converge. Learning rate of 0.01 improved the grading score to 40% and above.*

![Learning Rate ][image_9]

The optimal hyperparameters:

learning_rate = 0.001

batch_size = 100

num_epochs = 200

steps_per_epoch = 50

validation_steps = 12

workers = 4


### Would this model and data work well for following another object (dog, cat, car, etc.) instead of a human and if not, what changes would be required.

This model was trained on people, however, it could be used to train on any other objects of interest. For example, it could be trained on images of horses or trucks,car,dog,cat. The model could conceivably be trained on any set of labelled data large enough to create a model that works with more categories it is necessary to collect and label images with enough examples for each class with different poses, distances and lighting conditions. With such dataset on hand it is then possible to train a new model using the same technique described here. 

### Future Enhancements
Since there is a lot of room to improve the grade score  and training effeciency. A lot of good quality data needs to be generated and More convolutional layers can be added.

### CONCLUSION

This was an exciting project and explains in depth about how to train a deep learning neural network.Also how important is to collect good data. Good data is just as important as a good network architecture, so collecting the best data you can is a key to success!
