import tensorflow as tf
rg = tf.random.Generator.from_seed(1, alg='philox')


def preprocessing_fn(phases=5,roll=1,selected_image2=-1):
  """Returns a preprocessing function for training."""
  
  def _preprocessing_fn(zfill,image):
    """Preprocess the data.
    Roll and select nphases from zero filled images and image $selected_image2 (last image for lowest latency) 
    Args:
      inputs: Input data. A dict containing the following keys:
        - 'zfill': Zero filled images A tensor of shape [height, width,time].
        - 'image': Ground truth images A tensor of shape [height, width,time]
    Returns:
      A tuple (zerofilled image, ground truth image).
    """
    if roll>0:
        shift_im=rg.uniform(shape=(), minval=0, maxval=tf.shape(image)[-1]-phases, dtype=tf.int32)
        image=tf.roll(image,shift=-shift_im,axis=-1)
        zfill=tf.roll(zfill,shift=-shift_im,axis=-1)
      
    image=image[...,:phases]
    zfill=zfill[...,:phases]
    
    if selected_image2 is not None: 
      image=tf.expand_dims(image[...,selected_image2],axis=-1)
    #get rid of nans
    image = tf.where(tf.math.is_nan(image), 0., image)
    zfill = tf.where(tf.math.is_nan(zfill), 0., zfill)
    return zfill, image  # input (features), output (labels)
      
  return _preprocessing_fn
