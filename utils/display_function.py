import tensorflow as tf
import tensorflow_mri as tfmr
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import numpy as np

def display_fn(complex_part='abs',selected_image=-1,concat_axis=-2,input_shape=None,output_shape=None):
  """Returns a display function for tensorboard Images:
  display_fn: A callable. A function which accepts three arguments
      (features, labels and predictions for a single example) and returns the
      image to be written to TensorBoard. Overrides the default function, which
      concatenates selected features, labels and predictions according to
      `concat_axis`, `feature_keys`, `label_keys`, `prediction_keys` and
      `complex_part`."""
  
  def _display_fn(features, labels, predictions):
    """Returns the image to be displayed for each example.
    By default, the image is created by concatenating horizontally `features`,
    `labels` and `predictions`.
    The input is converted to complex first in this case.
    Args:
      features: Features (model inputs for a single example).
      labels: Labels (ground truth for a single example).
      predictions: Predictions (model outputs for a single example).
    Returns:
      The image to display.
    """
    if input_shape=='complex':
      features=_channels2complex(features)
      features=tf.expand_dims(features[...,selected_image],axis=-1)
    elif input_shape=='multicoil':
      features=_channels2complex(features)
      features=tf.sqrt(tf.reduce_sum(features*tf.math.conj(features),axis=-1,keepdims=True))
    elif input_shape=='varnet':
      features=_channels2complex(features[0])
    else:
      features=tf.complex(tf.expand_dims(features[...,selected_image],axis=-1),tf.zeros(labels.shape))
    if output_shape=='complex':
      labels=_channels2complex(labels)
      predictions=_channels2complex(predictions)
    else:
      labels=tf.complex(labels,tf.zeros(labels.shape))
      predictions=tf.complex(predictions,tf.zeros(predictions.shape))
      
    # Independently concatenate individual features, labels and predictions.
    cat_features = _select_and_concatenate(
        features, None, concat_axis, complex_part,
        arg_name='features')
    cat_labels = _select_and_concatenate(
        labels, None, concat_axis, complex_part,
        arg_name='labels')
    cat_predictions = _select_and_concatenate(
        predictions, None, concat_axis, complex_part,
        arg_name='predictions')

    # Concatenate features, labels and predictions.
    tensors = []
    if cat_features is not None:
      tensors.append(cat_features)
    if cat_labels is not None:
      tensors.append(cat_labels)
    if cat_predictions is not None:
      tensors.append(cat_predictions)
    if tensors:
      return tf.concat(tensors, concat_axis)

    return None


  def _select_and_concatenate(arg, keys, axis, complex_part, arg_name=None): 
    """Selects and concatenates the tensors for the given keys."""
    if not isinstance(arg, (tuple, dict, tf.Tensor)):
      raise TypeError(
          f"`{arg_name}` must be a tensor, tuple or dict, got: {arg}.")

    # Select specified values and concatenate them.
    if isinstance(arg, (tuple, dict)):
      if keys is None:
        tensors = list(arg.values()) if isinstance(arg, dict) else arg
      else:
        tensors = [arg[key] for key in keys]
      if not tensors:
        return None
      for index, tensor in enumerate(tensors):
        tensors[index] = _prepare_for_concat(tensor, complex_part)
      out = tf.concat(tensors, axis)
    else:  # Input is a tensor, so nothing to select/concatenate.
      out = _prepare_for_concat(arg, complex_part)

    return out


  def _prepare_for_concat(tensor, complex_part):  # pylint: disable=missing-param-doc
    """Prepares a tensor for concatenation."""
    if tensor is None:
      return None
    # If tensor is complex, convert to real.
    if tensor.dtype.is_complex:
      if complex_part is None:
        raise ValueError(
            "`complex_part` must be specified for complex inputs.")
      tensor = tfmr.image_ops.extract_and_scale_complex_part(
          tensor, complex_part, max_val=1.0)
    # Cast to common type (float32).
    return tf.cast(tensor, _CONCAT_DTYPE)

  def _channels2complex(data):
    # Convert 2 channel Real Imaginary to complex
    if not data.shape[-1] % 2 == 0:
            raise ValueError("Invalid input: Number of channels must be even. "
                             "Found {} channels.".format(data.shape[-1]))
    data_real = data[..., :data.shape[-1] // 2]
    data_imag = data[..., data.shape[-1] // 2:]
    data = tf.complex(data_real, data_imag)
    return data

  _CONCAT_DTYPE = tf.float32
  return _display_fn


def plotVid(imgx,axis=2,title='',savepath=None,vmin=None,vmax=None,interval=200,cmap='gray',overlay=None,figsize=(15,5),bg_color='w'):

    
    sha=imgx.shape
    plt.rc('animation', html='html5')
    fig = plt.figure(figsize=figsize, facecolor=bg_color) # make figure
    if bg_color=='k':
        fig.suptitle(title, fontsize=13,color='w')
        
    else:
        fig.suptitle(title, fontsize=13)
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ims=[]
    if overlay is not None:
        import dlex
        imgx=dlex.ops.image_ops.label_to_rgb(overlay, imgx, bg_label=0)

    if axis==2:
        for i in range(sha[axis]):
            ims.append(imgx[:,:,i])
    elif axis==0:
        for i in range(sha[axis]):
            ims.append(imgx[i,:,:])
        
    elif axis==1:
        for i in range(sha[axis]):
            ims.append(imgx[:,i,:])
    imagelist=ims
    
    if vmax is None:
        vmax=np.max(imagelist[:][:])*0.8
    if vmin is None:
        vmin=np.min(imagelist[:][:])
    im = plt.imshow(imagelist[0], cmap=plt.get_cmap(cmap), vmin=vmin, vmax=vmax)
    
    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(imagelist[j])
        # return the artists set
        return [im]
    # kick off the animation
    animation2 = ani.FuncAnimation(fig, updatefig, frames=range(sha[axis]),interval=interval, blit=True)
    plt.show()
    
    if savepath is not None:
        if savepath.endswith('.gif'):
            animation2.save((savepath))
        else:
            animation2.save((savepath + '.mp4'))
    return animation2