import tensorflow as tf
rg = tf.random.Generator.from_seed(1, alg='philox')

def preprocessing_fn(phases=24,roll=1,rotation=0,input_format= None):
  """Returns a preprocessing function for training."""
  
  def _preprocessing_fn(zfill,image):
    """Preprocess the data.
    Roll and select nphases from zero filled images and image $selected_image2 (last image for lowest latency) 
    Args:
      inputs: Input data. A dict containing the following keys:
        - 'zfill': Zero filled images A tensor of shape [height, width,time (,channels)].
        - 'image': Ground truth images A tensor of shape [height, width,time (,channels)]
    Returns:
      A tuple (zerofilled image, ground truth image).
    """
    
    if tf.rank(zfill)==3:
      zfill=tf.expand_dims(zfill,axis=-1)
    zfill=tf.transpose(zfill,perm=[2,0,1,3])
    image=tf.transpose(tf.expand_dims(image,axis=-1),perm=[2,0,1,3])
    
    if rotation>0:
      rot_im=rg.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
      image=tf.image.rot90(image,k=rot_im)
      zfill=tf.image.rot90(zfill,k=rot_im)
      
    if roll>0:
        shift_im=rg.uniform(shape=(), minval=0, maxval=tf.shape(image)[0]-phases, dtype=tf.int32)
        image=tf.roll(image,shift=-shift_im,axis=0)
        zfill=tf.roll(zfill,shift=-shift_im,axis=0)
    image=image[:phases,...]
    zfill=zfill[:phases,...]

    #get rid of nans
    image = tf.where(tf.math.is_nan(image), 0., image)
    zfill = tf.where(tf.math.is_nan(zfill), 0., zfill)

    return zfill, image  # input (features), output (labels)
      
  return _preprocessing_fn
