Investigating the use of high spatio-temporal resolution publicly available natural videos to learn Dynamic MR image reconstruction
====================================================================================================================================

Synopsis: 
---------

Images from Inter4K \[1\]: a high spatio-temporal resolution publicly available natural video dataset 
are used to learn Dynamic MR image reconstruction.

Methods can be found in \[link\].

Three networks and trajectories were investigated: 
- Cartesian real-time with unrolled VarNet \[2\]
- Radial real-time with multicoil UNet
- Low latency spiral imaging with FastDVDNet (Hyperslice \[3\]) 

https://github.com/olivier-jaubert/Image_Reconstruction_Inter4k/assets/68073827/be038e26-7f36-4ff6-8b40-b6e0bdf47c11

- Top Line: RGB Video, Undersampled Cartesian, Undersampled Radial, Undersampled Spiral.

- Bottom Line: Target,    VarNet Reconstruction, multicoil 3DUNet Reconstruction, FastDVDNet Reconstruction.


Provided code includes trajectories, model training and pre-trained models as implemented for the paper.

The ethics does not allow sharing medical image data therefore only Inter4K data and models are made available. 

------------------------------------------------------

Installation
============

For installation please:
1) Download github repository.

2) From Project folder, create Docker image and launch interactive docker container: 
```
docker compose up --build -d
```   

3) Download and unzip Inter4K Dataset in DatasetFolder (see DatasetFolder/README.md if does not work):
```
docker compose exec tensorflow python download_Inter4k_Dataset.py
```

3) Test training by using one of the following commands :

```
docker compose exec tensorflow python train\_network.py -m VarNet # for VarNet Cartesian training (longest)
docker compose exec tensorflow python train\_network.py -m 3DUNet # for full model 3DUNet radial training
docker compose exec tensorflow python train\_network.py -m FastDVDNet # for FastDVDNet Spiral training
```


4) Can be used with VScode (.devcontainer folder) for development within the docker container.

Note that only Linux is supported.

Results are saved in ./Training\_folder (as in the already trained example models ./Training\_folder/Default\_FastDVDNet)

Acknowledgments
===============

\[1\] Stergiou, A., & Poppe, R. (2023). AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling. IEEE Transactions on Image Processing, 32, 251–266. https://doi.org/10.1109/TIP.2022.3227503

\[2\] Hammernik, K., Klatzer, T., Kobler, E., Recht, M. P., Sodickson, D. K., Pock, T., & Knoll, F. (2018). Learning a variational network for reconstruction of accelerated MRI data. Magnetic Resonance in Medicine, 79(6), 3055–3071. https://doi.org/10.1002/mrm.26977

\[3\] Jaubert, O., Montalt-Tordera, J., Knight, D., Arridge, S., Steeden, J., & Muthurangu, V. (2023). HyperSLICE: HyperBand optimized spiral for low-latency interactive cardiac examination. Magnetic Resonance in Medicine. https://doi.org/10.1002/MRM.29855
