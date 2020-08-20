# Speckle-Denoise
Deep Learning Approaches on Medical Ultrasound Image Denoising in Pytorch and Tensorflow.

# Objectives
Medical ultrasound is becoming today one of the most accessible diagnostic imaging modalities. A high image quality is the basis on which clinical interpretation can be made with sufficient confidence. However, medical ultrasound images suffer typically from speckle effect due to interference in the image formation.

we propose to study deep learning approaches to perform ultrasound speckle reduction. We will explore several different methods including end-to-end learning and hybrid methods. We would also like to push the understanding and the interpretation of the network behavior through a deeper analysis of network structures and activation functions.

# State-of-the-art algorithms
### Models
 * DnCNN [[PDF]](https://arxiv.org/pdf/1608.03981v1.pdf) [[Web]](https://github.com/cszn/DnCNN) 
   * Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (TIP2017), Zhang et al.
 * FFDNet [[PDF]](https://arxiv.org/pdf/1710.04026.pdf) [[Web]](https://github.com/cszn/FFDNet) 
   * FFDNet: Toward a fast and flexible solution for CNN-based image denoising, Zhang et al.
 * CBDNet [[PDF]](https://arxiv.org/pdf/1807.04686.pdf) [[WEB]](https://github.com/GuoShi28/CBDNet)
   *  Toward Convolutional Blind Denoising of Real Photographs (Arxiv2018), Shi Guo, Zifei Yan, Kai Zhang, Wangmeng Zuo, Lei Zhang.
 * MWCNN [[PDF]](https://arxiv.org/pdf/1805.07071.pdf) [[WEB]](https://github.com/lpj0/MWCNN)
   *  Multi-level wavelet-CNN for image restoration (2018), Liu, Pengju, et al.
 * KPN [[PDF]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/3761.pdf)
   * Burst Denoising With Kernel Prediction Networks (CVPR2018), Ben Mildenhall, Jonathan T. Barron, Jiawen Chen, Dillon Sharlet, Ren Ng, Robert Carroll.
   
### Generalization Capacity
 * CBDNet [[PDF]](https://arxiv.org/pdf/1807.04686.pdf) [[WEB]](https://github.com/GuoShi28/CBDNet)
   *  Toward Convolutional Blind Denoising of Real Photographs (Arxiv2018), Shi Guo, Zifei Yan, Kai Zhang, Wangmeng Zuo, Lei Zhang.
 * Model without bias [[PDF]](https://arxiv.org/pdf/1906.05478.pdf)
   *  Robust and interpretable blind image denoising via bias-free convolutional neural networks, Mohan, Sreyas, et al.
   
# Codes
### Models
- [model_baseline.py] U-net
- [model_gdfn2.py] global dynamic filter network
- [model_local_dfn.py] local dynamic filter network
- [model_mwcnn.py] multi-level wavelet cnn
- [model_kpn.py] kernel predicion network
- [model_mwkpn.py] multi-level wavelet kernel prediction network

### train
- [train_blind_noise.ipynb] train models without noise estimation
- [train_burst_blind_noise.ipynb] train burst models without noise estimation
- [train_burst_with_noise_est.ipynb] train burst models with noise estimation
- [train_noise_est.ipynb] train noise estimation models
- [train_ultrasound_data.ipynb] train ultrasound data with one of the above models

### test
- [test_ultrasound_data.ipynb] train ultrasound data with one of the above models

### eval
- [eval_gdfn.ipynb] evaluate global dfn filters
- [eval_kpn.ipynb] evaluate kpn/local dfn filters
- [eval_mwcnn.ipynb] evaluate multi-level wavelet effects on U-net
- [eval_transfer_learning.ipynb] evaluate transfer learning effects

### method
- [dwt.py] discrete wavelet transform implementation in tensorflow2
- [hybrid_method.ipynb] hybride methode implementation

   
# Notes
- Codes in Juppyter Notebook are for educative demonstration. 
- You can also consult the **[Project Report](https://github.com/chenqianben/Speckle-Denoise/blob/master/Project%20Report%20-%20FR.pdf)** for more information. 
