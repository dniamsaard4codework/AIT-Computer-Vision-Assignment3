# Assignment 3

**Due Date:** Nov 16, 2025

**How to submit**:

To avoid file size limitation,

- push all codes (jupyter notebooks) to your github repo
- submit the link to your repo.

## Task 1: Graph Cut Segmentation

**Instructions:**

- Given 2 images (asm-1, asm2), generate a bounding box for a person in each images by utilizing any deep learning-based object detectors (pretrain model).
- Leveraging the generated bounding boxes, implement graph-based image segmentation using OpenCV function, `cv2.grabCut`.
- Run GrabCut for 1, 3, and 5 iterations — report qualitative and quantitative differences.
- Visualize Results: Display original images, user masks, and final segmentation results (foreground only, and overlay).

**Deliverable:**

- Jupyter notebook with clear code and comments.

---------------------------------------------------------------
## Task 2: Fully Convolutional Network (FCN)

Implement an FCN for semantic segmentation, train it on a small dataset, and analyze how architectural and training choices affect performance.

**Instructions:**

**Dataset**:
- Use a subset of Pascal VOC, COCO, or a small custom dataset (10–20 images for quick runs).
- Split into train/test.
- Preprocess (resize, normalize, and convert masks to class indices).
**Model Implementation**:
- Implement FCN-32s, FCN-16s, or FCN-8s variants.
- Use pretrained ResNet/VGG (remove final FC layers) as a backbone
- Upsampling Method: 1) transpose convolution vs 2) bilinear interpolation.
**Training**
- Loss: CrossEntropyLoss
- Optimizer: Adam or SGD
- Metrics: Mean IoU, pixel accuracy
- Train for 20 epochs (or until convergence).
- Log training curves.

**Deliverable:**

- Jupyter notebook with clear code and comments.
- Visualization of segmentation results (min. 3 test images).
- Table comparing transpose convolution vs bilinear interpolation.
- Summarize with visuals and short analysis.

---------------------------------------------------------------
## Task-3: Variational Autoencoder (VAE)

Implement a VAE to learn latent representations of images, generate new samples, and analyze the effect of latent dimensionality

**Instructions:**

**Dataset:** Use MNIST, and preprocess
**Model Implementation**:
- Encoder: 3–4 Conv layers → flatten → Linear → output μ and log(σ²)
- Use latent dimension of 128
- Decoder: transpose convolutions → reconstruct image.
**Training**
- Loss: Reconstruction (MSE or BCE), KL divergence term
- Optimizer: Adam
- Train for 50 epochs.

- Visualize reconstruction and latent space evolution.
- Generation and Visualization: Sample random z vectors → generate images.
- Interpolate between two z vectors to observe smooth transitions.
- Change latent dimension to 256, retrain the model, and visualize generated images and reconstruction quality.

**Deliverable:**

- Jupyter notebook with clear code and comments.