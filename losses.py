import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
regression loss 
"""


class HuberLoss(keras.losses.Loss):
    """Huber loss for regression problem
    """

    def __init__(self, threshold=1.0, **kwargs):
        """
        Args:
            threshold (float): threshold value to define whether is a small error. Defaults to 1.0.
        """
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        Args:
            y_true (tensor): Ground truth values. shape = [batch_size, d0, .. dN].
            y_pred (tensor): The predicted values. shape = [batch_size, d0, .. dN].

        Returns:
            (scalar): Mean squared error values. shape = [batch_size, d0, .. dN-1].
        """
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


# cross entropy with weights
class CrossEntropyWeightedLoss(keras.losses.Loss):
    """weighted cross entropy for multi-class
    """

    def __init__(self, weights, **kwargs):
        """
        Args:
            weights (list): weights for each class, e.g. [0.5, 0.2, 0.3]
            num_classes (int): the number of classes
        """
        
        self.weights = np.array(weights)/sum(weights)
        self.num_classes = len(weights)
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        Args:
            labels (tensor): Ground truth values. shape = [batch_size, d0, .. dN], e.g. [B,N,].
            logits (tensor): The predicted values(not probability form). shape = [batch_size, d0, .. dN,K], e.g. [B,N,K]

        Returns:
            (scalar): Mean weighted cross entropy values of each feature.
        """

        y_true = tf.one_hot(y_true, depth=self.num_classes)  # [B,N,K].
        weights_of_sample = tf.reduce_sum(
            self.weights * y_true, axis=-1)  # [B,N]
        loss_original = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred)  # [B,N]
        output_loss = tf.reduce_mean(
            loss_original * weights_of_sample)  # scalar
        return output_loss

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "weights": self.weights
        }

# dice loss and variants(log-cosh dice loss)
class DiceLoss(keras.losses.Loss):
    """Dice loss for segmentation task
    """

    def __init__(self, eps=1e-8,log_cosh=False, **kwargs):
        """
        Args:
            eps (float): used for avoiding division by zero error
            log_cosh (bool): use logh_cosh dice loss or not, check: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
        """
        self.eps = eps
        self.log_cosh = log_cosh
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        Args:
            y_true (tensor): Ground truth values. shape = [batch_size, d0, .. dN], e.g. [B,N,].
            y_pred (tensor): The predicted values(not probability form). shape = [batch_size, d0, .. dN,K], e.g. [B,N,K]

        Returns:
            (scalar): Mean weighted cross entropy values of each feature.
        """
        # create one hot form of labels
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(y_true, depth=num_classes) # B,N,K
        input_soft= tf.nn.softmax(y_pred, axis=-1)# B,N,K

        # compute the actual dice score
        intersection = tf.reduce_sum(input_soft * y_true, axis=-1)
        cardinality = tf.reduce_sum(input_soft + y_true, axis=-1)
        dice_score = 2. * intersection / (cardinality + self.eps)
        dice_loss = tf.reduce_mean(-dice_score + 1.)
        
        if self.log_cosh:
            return tf.math.log((tf.exp(dice_loss) + tf.exp(-dice_loss)) / 2.0)
        else:
            return dice_loss

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "eps": self.eps,
            "log_cosh": self.log_cosh
        }


# focal loss and variants(focal tvesky loss)
class FocalLoss(keras.losses.Loss):
    """Dice loss for segmentation task
    """

    def __init__(self,alpha=0.5,gamma=2.0,eps=1e-8, **kwargs):
        """
        Args:
            eps (float): used for avoiding division by zero error
        """
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        Args:
            y_true (tensor): Ground truth values. shape = [batch_size, d0, .. dN], e.g. [B,N,].
            y_pred (tensor): The predicted values(not probability form). shape = [batch_size, d0, .. dN,K], e.g. [B,N,K]

        Returns:
            (scalar): Mean weighted cross entropy values of each feature.
        """
        # create one hot form of labels
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(y_true, depth=num_classes) # B,N,K
        input_soft= tf.nn.softmax(y_pred, axis=-1) + self.eps# B,N,K

        # compute the actual focal score
        weight = tf.pow(-input_soft + 1., self.gamma)
        focal = -self.alpha * weight * tf.math.log(input_soft)
        loss_tmp = tf.reduce_sum(y_true * focal, axis=-1)

        return tf.reduce_mean(loss_tmp)
        

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps": self.eps
        }

# tversky loss
class TverskyLoss(keras.losses.Loss):
    """Tversky loss for segmentation task
    """

    def __init__(self,alpha=0.5,beta=0.5,eps=1e-8, **kwargs):
        """
        Args:
            eps (float): used for avoiding division by zero error
        """
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        Args:
            y_true (tensor): Ground truth values. shape = [batch_size, d0, .. dN], e.g. [B,N,].
            y_pred (tensor): The predicted values(not probability form). shape = [batch_size, d0, .. dN,K], e.g. [B,N,K]

        Returns:
            (scalar): Mean weighted cross entropy values of each feature.
        """
        # create one hot form of labels
        num_classes = tf.shape(y_pred)[-1]
        y_true = tf.one_hot(y_true, depth=num_classes) # B,N,K
        input_soft= tf.nn.softmax(y_pred, axis=-1)# B,N,K

        # compute the actual dice score
        intersection = tf.reduce_sum(input_soft * y_true, axis=-1)
        fps = tf.reduce_sum(input_soft * (-y_true + 1.), axis=-1)
        fns = tf.reduce_sum((-input_soft + 1.) * y_true, axis=-1)
        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)

        return tf.reduce_mean(-tversky_loss + 1.)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "eps": self.eps
        }


# lovasz loss
class LovaszLoss(keras.losses.Loss):
    """Lovasz loss for segmentation task
    """

    def __init__(self, **kwargs):
        """
        Args:
            eps (float): used for avoiding division by zero error
        """
        # self.eps = eps
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """

        Args:
            y_true (tensor): Ground truth values. shape = [batch_size, d0, .. dN], e.g. [B,N,].
            y_pred (tensor): The predicted values(not probability form). shape = [batch_size, d0, .. dN,K], e.g. [B,N,K]

        Returns:
            (scalar): Mean weighted cross entropy values of each feature.
        """
        # convert probs
        input_soft= tf.nn.softmax(y_pred, axis=-1)# B,N,K
        return lovasz_softmax(input_soft,y_true)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            # "eps": self.eps
        }

"""
lovasz implementation from Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    # if len(probas.shape) == 3:
    #     probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[-1]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

# combound loss (e.g. CE+ dice loss)
