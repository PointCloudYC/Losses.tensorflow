# Losses in Tensorflow

## losses implemented

**losses all inherit from kereas.losses.loss**

- cross entropy with weights

![](images/ce_loss.jpg)

- dice loss
- focal loss
- tversky loss
- lovasz-softmax loss

## requirments

tensorflow 1.15+ (also suitable for tensorflow 2.x)

Note: not supported by tensorflow 1.13 or older version since keras.losses.Loss is not defined.

## how to use?

check an example for dice loss, other losses are similar, for more examples check `main.py`
```
logits_all_correct = [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [4.0, 8.0, 3.0],[4.0, 8.0, 9.0]]
logits_all_wrong = [[2.0, 4.0, 1.0], [5.0, 2.0, 1.0], [8.0, 4.0, 3.0],[8.0, 2.0, 2.0]]
labels = [0, 1, 1, 2]

loss_fn = DiceLoss()
loss_fn(y_true=labels,y_pred=logits_all_correct),

print("Dice losses are {:.3f},{:.3f},{:.3f}".format(
    loss_name,
    loss_fn(y_true=labels,y_pred=logits_all_correct),
    loss_fn(labels,logits_all_wrong),
))
```

for training, just pass this loss to fit or relevant methods

```
import losses

# declare the model
loss_fn = losses.DiceLoss()
model.compile(loss=loss_fn, optimizer='adam')
...
```

## refs
* [shruti-jadon/Semantic-Segmentation-Loss-Functions: This Repository is implementation of majority of Semantic Segmentation Loss Functions](https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions)
* [JunMa11/SegLoss: A collection of loss functions for medical image segmentation](https://github.com/JunMa11/SegLoss)