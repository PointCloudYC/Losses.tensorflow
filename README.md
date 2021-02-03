# Losses in Tensorflow

## implemented losses

**losses inherit from kereas.losses.loss**

- cross entropy with weights
![](images/ce_loss.jpg)

- dice loss
- focal loss
- tversky loss
- lovasz-softmax loss

## requirments

tensorflow 1.15+ (also suitable for tensorflow 2.x)

Note: not supported by tensorflow 1.13 or older version since keras.losses.Loss is not defined.

## refs
* [shruti-jadon/Semantic-Segmentation-Loss-Functions: This Repository is implementation of majority of Semantic Segmentation Loss Functions](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions)
* [JunMa11/SegLoss: A collection of loss functions for medical image segmentation](https://github.com/JunMa11/SegLoss)