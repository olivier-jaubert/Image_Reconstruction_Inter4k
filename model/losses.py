import tensorflow as tf
import tensorflow_mri as tfmri
from tensorflow_mri.python.losses import iqa_losses
from tensorflow_mri.python.util import keras_util


@tf.keras.utils.register_keras_serializable(package="MRI")
def mean_squared_error(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  return tf.math.reduce_mean(
      tf.math.real(tf.math.squared_difference(y_pred, y_true)), axis=-1)


@tf.keras.utils.register_keras_serializable(package="MRI")
def mean_absolute_error(y_true, y_pred):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  return tf.math.reduce_mean(tf.math.abs(y_pred - y_true), axis=-1)


@tf.keras.utils.register_keras_serializable(package="MRI")
def mean_absolute_gradient_error(y_true, y_pred, method='sobel',
                                 norm=False, batch_dims=None, image_dims=None):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)

  grad_true = tfmri.image.image_gradients(
      y_true, method=method, norm=norm,
      batch_dims=batch_dims, image_dims=image_dims)
  grad_pred = tfmri.image.image_gradients(
      y_pred, method=method, norm=norm,
      batch_dims=batch_dims, image_dims=image_dims)

  return mean_absolute_error(grad_true, grad_pred)


@tf.keras.utils.register_keras_serializable(package="MRI")
def mean_squared_gradient_error(y_true, y_pred, method='sobel',
                                norm=False, batch_dims=None, image_dims=None):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)

  grad_true = tfmri.image.image_gradients(
      y_true, method=method, norm=norm,
      batch_dims=batch_dims, image_dims=image_dims)
  grad_pred = tfmri.image.image_gradients(
      y_pred, method=method, norm=norm,
      batch_dims=batch_dims, image_dims=image_dims)

  return mean_squared_error(grad_true, grad_pred)


@tf.keras.utils.register_keras_serializable(package="MRI")
class MeanSquaredError(keras_util.LossFunctionWrapper):
  def __init__(self,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='mean_squared_error'):
    super().__init__(mean_squared_error, reduction=reduction, name=name)


@tf.keras.utils.register_keras_serializable(package="MRI")
class MeanAbsoluteError(keras_util.LossFunctionWrapper):
  def __init__(self,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='mean_absolute_error'):
    super().__init__(mean_absolute_error, reduction=reduction, name=name)


@tf.keras.utils.register_keras_serializable(package="MRI")
class MeanAbsoluteGradientError(iqa_losses.LossFunctionWrapperIQA):
  def __init__(self,
               method='sobel',
               norm=False,
               batch_dims=None,
               image_dims=None,
               multichannel=True,
               complex_part=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='mean_absolute_gradient_error'):
    super().__init__(mean_absolute_gradient_error,
                     reduction=reduction, name=name, method=method,
                     norm=norm, batch_dims=batch_dims, image_dims=image_dims,
                     multichannel=multichannel, complex_part=complex_part)


@tf.keras.utils.register_keras_serializable(package="MRI")
class MeanSquaredGradientError(iqa_losses.LossFunctionWrapperIQA):
  def __init__(self,
               method='sobel',
               norm=False,
               batch_dims=None,
               image_dims=None,
               multichannel=True,
               complex_part=None,
               reduction=tf.keras.losses.Reduction.AUTO,
               name='mean_squared_gradient_error'):
    super().__init__(mean_squared_gradient_error,
                     reduction=reduction, name=name, method=method,
                     norm=norm, batch_dims=batch_dims, image_dims=image_dims,
                     multichannel=multichannel, complex_part=complex_part)


# @tf.keras.utils.register_keras_serializable(package="MRI")
# class MAEPlusMSSSIM(keras_util.LossFunctionWrapper):
#   def __init__(self,
#                reduction=tf.keras.losses.Reduction.AUTO,
#                name='mae_plus_msssim'):
#     super().__init__(lib.mae_plus_msssim, reduction=reduction, name=name)


@tf.keras.utils.register_keras_serializable(package="Library")
class WeightedSumLoss(tf.keras.losses.Loss):
  def __init__(self,
               losses,
               weights,
               reduction=tf.keras.losses.Reduction.AUTO,
               name=None):
    self.losses = [tf.keras.losses.get(loss) for loss in losses]
    self.weights = weights
    name = '_p_'.join(
        [loss.name if isinstance(loss, tf.keras.losses.Loss) else
         loss.__name__ for loss in self.losses])
    super().__init__(reduction=reduction, name=name)

  def call(self, y_true, y_pred):
    loss = self.losses[0](y_true, y_pred) * self.weights[0]
    for loss_fn, weight in zip(self.losses[1:], self.weights[1:]):
      loss += loss_fn(y_true, y_pred) * weight
    return loss

  def get_config(self):
    base_config = super().get_config()
    config = {
        'losses': [tf.keras.losses.serialize(loss) for loss in self.losses],
        'weights': self.weights
    }
    return {**config, **base_config}