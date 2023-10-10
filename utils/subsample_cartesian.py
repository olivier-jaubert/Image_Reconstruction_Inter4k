"""
This source code is based on the fastMRI repository from Facebook AI
Research and is used as a general framework to handle MRI data. Link:

https://github.com/facebookresearch/fastMRI
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import random
@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int], half_fourier: False):
        """
        Args:
            center_fractions: When using a random mask, number of low-frequency
                lines to retain. When using an equispaced masked, fraction of
                low-frequency lines to retain.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )
        self.half_fourier = half_fourier
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) :
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration
        
        
        
class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a Cartesian sub-sampling mask of a given shape,
    as implemented in
    "A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image
    Reconstruction" by J. Schlemper et al.

    The mask selects a subset of rows from the input k-space data. If the
    k-space data has N rows, the mask picks out:
        1. center_fraction rows in the center corresponding to low-frequencies.
        2. The remaining rows are selected according to a tail-adjusted 
           Gaussian probability density function. This ensures that the
           expected number of rows selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None 
    ):
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the third
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")
            
        with temp_seed(self.rng, seed):
            sample_n, acc = self.choose_acceleration()
        
        N, Nc, Nx, Ny, Nch = shape
        Nfloat=tf.cast(Nx,tf.float32)
        Nfloat=Nx
        # generate normal distribution
        normal_pdf = lambda length, sensitivity: np.exp(-sensitivity * (np.arange(length) - length / 2)**2)
        pdf_x = normal_pdf(Nfloat, 0.5/(Nfloat/10.)**2)
        lmda = Nfloat / (2.*acc)
        n_lines = int(Nfloat / acc)
    
        # add uniform distribution so that probability of sampling
        # high-frequency lines is non-zero
        pdf_x += lmda * 1./Nfloat
    
        if sample_n:
            # lines are never randomly sampled from the already
            # sampled center
            # temp=tf.where(tf.range(Nx)>(Nx//2 + sample_n//2), tf.zeros(tf.shape(pdf_x),dtype=pdf_x.dtype),pdf_x)
            # temp2=tf.where(tf.range(Nx)<(Nx//2 - sample_n//2), tf.zeros(tf.shape(pdf_x),dtype=pdf_x.dtype),pdf_x)
            # pdf_x=temp+temp2
            if self.half_fourier>0:
                pdf_x[Nx//2 - sample_n//2 : Nx//2 + sample_n//2] = 0
                pdf_x[ : int(Nx*(1-self.half_fourier))] = 0

            else:
                pdf_x[Nx//2 - sample_n//2 : Nx//2 + sample_n//2] = 0
            #pdf_x /= tf.reduce_sum(pdf_x)  # normalise distribution
            pdf_x /= np.sum(pdf_x)  # normalise distribution
            n_lines2 = n_lines-sample_n
    
        mask = np.zeros((N, Nx))
        for i in range(N):
            # select low-frequency lines according to pdf
            idx = np.random.choice(Nx, n_lines2, False, pdf_x)
            mask[i, idx] = 1
        # #Do not reselect same lines
        # idx = np.random.choice(Nx, n_lines2*N, False, pdf_x)
        # indexes=np.reshape(idx,[n_lines2,N])
        # indexes=np.transpose(indexes)
        # for ii in range(N):
        #     mask[ii,indexes[ii,:]]=1
        if sample_n:
            # central lines are always sampled
            mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
        
        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-3] = Nx
        mask_shape[0] = N
        mask = mask.reshape(*mask_shape).astype(np.float32)
        return mask    

class NoReselectMaskFunc(MaskFunc):
    """
    NoReselectMaskFunc creates a Cartesian sub-sampling mask of a given shape.

    The mask selects a subset of rows from the input k-space data. If the
    k-space data has N rows, the mask picks out:
        1. center_fraction rows in the center corresponding to low-frequencies.
        2. The remaining rows are selected according to a uniform probability density function. This ensures that the
           expected number of rows selected is equal to (N / acceleration). Additionally cannot reselect same lines 
           for different cardiac phases

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the NoReselectMaskFunc object is called.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None 
    ):
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the third
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")
            
        with temp_seed(self.rng, seed):
            sample_n, acc = self.choose_acceleration()
        
        N, Nc, Nx, Ny, Nch = shape
        Nfloat=tf.cast(Nx,tf.float32)
        Nfloat=Nx
        # generate normal distribution
        # normal_pdf = lambda length, sensitivity: np.exp(-sensitivity * (np.arange(length) - length / 2)**2)
        # pdf_x = normal_pdf(Nfloat, 0.5/(Nfloat/10.)**2)
        uniform_pdf = lambda length: np.ones((length,),dtype=np.float32)
        pdf_x = uniform_pdf(Nx)
        lmda = Nfloat / (2.*acc)
        n_lines = int(Nfloat / acc)
    
        # add uniform distribution so that probability of sampling
        # high-frequency lines is non-zero
        pdf_x += lmda * 1./Nfloat
    
        if sample_n:
            # lines are never randomly sampled from the already
            # sampled center
            # temp=tf.where(tf.range(Nx)>(Nx//2 + sample_n//2), tf.zeros(tf.shape(pdf_x),dtype=pdf_x.dtype),pdf_x)
            # temp2=tf.where(tf.range(Nx)<(Nx//2 - sample_n//2), tf.zeros(tf.shape(pdf_x),dtype=pdf_x.dtype),pdf_x)
            # pdf_x=temp+temp2
            if self.half_fourier>0:
                pdf_x[Nx//2 - sample_n//2 : Nx//2 + sample_n//2] = 0
                pdf_x[ : int(Nx*(1-self.half_fourier))] = 0

            else:
                pdf_x[Nx//2 - sample_n//2 : Nx//2 + sample_n//2] = 0
            #pdf_x /= tf.reduce_sum(pdf_x)  # normalise distribution
            pdf_x /= np.sum(pdf_x)  # normalise distribution
            n_lines2 = n_lines-sample_n
    
        mask = np.zeros((N, Nx))
        # for i in range(N):
        #     # select low-frequency lines according to pdf
        #     idx = np.random.choice(Nx, n_lines2, False, pdf_x)
        #     mask[i, idx] = 1
        #Do not reselect same lines
        idx = np.random.choice(Nx, n_lines2*N, False, pdf_x)
        indexes=np.reshape(idx,[n_lines2,N])
        indexes=np.transpose(indexes)
        for ii in range(N):
            mask[ii,indexes[ii,:]]=1
        if sample_n:
            # central lines are always sampled
            mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
        
        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-3] = Nx
        mask_shape[0] = N
        mask = mask.reshape(*mask_shape).astype(np.float32)
        return mask

class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N rows, the mask picks out:
        1. N_low_freqs = (N * center_fraction) rows in the center
           corresponding to low-frequencies.
        2. The other rows are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of rows selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ):
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the third last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_rows = shape[-3]
            num_low_freqs = int(round(num_rows * center_fraction))

            # create the mask
            mask = np.zeros(num_rows, dtype=np.float32)
            pad = (num_rows - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_rows)) / (
                num_low_freqs * acceleration - num_rows
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_rows - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-3] = num_rows
            mask = mask.reshape(*mask_shape).astype(np.float32)

        return mask

def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    half_fourier: False,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations,half_fourier)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "noreselect":
        return NoReselectMaskFunc(center_fractions, accelerations,half_fourier)
    else:
        raise Exception(f"{mask_type_str} not supported")



#From TensorflowMRI
import tensorflow as tf   
def random_sampling_mask(shape, density=1.0, seed=None, rng=None, name=None):
  """Returns a random sampling mask with the given density.

  Args:
    shape: A 1D integer `Tensor` or array. The shape of the output mask.
    density: A `Tensor`. A density grid. After broadcasting with `shape`,
      each point in the grid represents the probability that a given point will
      be sampled. For example, if `density` is a scalar, then each point in the
      mask will be sampled with probability `density`. A non-scalar `density`
      may be used to create variable-density sampling masks.
      `tfmri.sampling.density_grid` can be used to create density grids.
    seed: A `Tensor` of shape `[2]`. The seed for the stateless RNG. `seed` and
      `rng` may not be specified at the same time.
    rng: A `tf.random.Generator`. The stateful RNG to use. `seed` and `rng` may
      not be specified at the same time. If neither `seed` nor `rng` are
      provided, the global RNG will be used.
    name: A name for this op.

  Returns:
    A boolean tensor containing the sampling mask.

  Raises:
    ValueError: If both `seed` and `rng` are specified.
  """
  with tf.name_scope(name or 'sampling_mask'):
    if seed is not None and rng is not None:
      raise ValueError("Cannot provide both `seed` and `rng`.")
    counts = tf.ones(shape, dtype=density.dtype)
    if seed is not None:  # Use stateless RNG.
      mask = tf.random.stateless_binomial(shape, seed, counts, density)
    else:  # Use stateful RNG.
      rng = rng or tf.random.get_global_generator()
      mask = rng.binomial(shape, counts, density)
    return tf.cast(mask, tf.bool)
  
def generate_mask(seed_value,data_shape,accelerations=[14],center_fractions=[4],half_fourier=0,mask_type='random'):
    #Set seed for all packages
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    data=np.ones(data_shape)
    shape = np.array(data.shape)
    #accelerations=[14]
    #center_fractions=[4]
    mask_func=create_mask_for_mask_type(mask_type,center_fractions=center_fractions,accelerations=accelerations,half_fourier=half_fourier)

    shape[1] = 1    # Each coil has the same type of mask
    mask = mask_func(shape,0)
    #masked_data = data * mask + 0.0 
    #print(mask.shape,np.sum(mask,axis=(1,2,3,4)))#number of lines sampled
    return mask