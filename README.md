Learning Dynamic MRI from publicly available Natural Videos
======================================================================================

Synopsis: 
---------

Three networks and trajectories were investigated: 
- Cartesian real-time with unrolled VarNet [ref]
- Radial real-time with multicoil UNet [ref]
- Low latency spiral imaging with FastDVDNet (Hyperslice [ref]) 
Provided code includes trajectories, model training and pre-trained models as implemented for the paper.

The ethics does not allow sharing medical image data therefore only Inter4K data and models are made available. 

------------------------------------------------------

Installation
============

All required packages can be installed using conda in a virtual environment:

``` {.console}
$ conda env create --name hyperslice --file environment.yml
```

Note that only Linux is supported.

Training and testing
====================

Once the environment created, activate using:

``` {.console}
$ conda activate hyperslice
```

You can then run:

-   the python file example.py file from project directory.

``` {.console}
$ python example.py
```

-   or the ipython notebook example.ipynb (to see intermediate results)

Results are saved in ./Training\_folder (as in the already trained
exemple model ./Training\_folder/Exemple\_Trained\_FastDVDnet)

Logs can be visualised using tensorboard:

``` {.console}
$ tensorboard --logdir path_to_directory
```

Acknowledgments
===============

Network architecture inspired from original FastDVDnet \[1\].
Code relies heavily on TensorFlow \[2\] and TensorFlow\_MRI \[3\].

\[1\] ​​Tassano, M., Delon, J., & Veit, T. (2020). FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation. 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 1351–1360.

\[2\] Abadi M, Barham P, Chen J, et al. TensorFlow: A System for Large-Scale Machine Learning TensorFlow: A system for large-scale machine learning. Proceedings of the 12th USENIX Symposium on Operating Systems Design and Implementation 2016:265--283.

\[3\] Montalt Tordera J. TensorFlow MRI. 2022 doi:10.5281/ZENODO.7120930.
https://pypi.org/project/tensorflow-mri/
