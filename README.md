# AIT-Computer-Vision-Assignment3

This assignment explores three fundamental tasks in computer vision, progressing from classic algorithms combined with deep learning to advanced generative models. Each task is a self-contained experiment analyzing the performance of different architectures and parameters.

---

## Task 1: Graph Cut Segmentation

### Objective

To segment a person from the background in two images (`asm-1` and `asm-2`). This task used a hybrid approach:

1.  **Object Detection:** A pre-trained Faster R-CNN model was used to automatically find the person and generate a bounding box.
2.  **Segmentation:** The OpenCV `cv2.grabCut` function was initialized with this bounding box to create a precise foreground mask.

The experiment analyzed the effect of running GrabCut for 1, 3, and 5 iterations.

### Summary of Results

The number of iterations had a significant and varied impact on segmentation quality.

| Image | 1 Iteration (Foreground Pixels) | 3 Iterations (Foreground Pixels) | 5 Iterations (Foreground Pixels) |
| :--- | :--- | :--- | :--- |
| **asm-1 (Cyclist)** | 13,260 | 13,158 | 13,111 |
| **asm-2 (Skater)** | 21,324 | 25,478 | 14,097 |

* **For `asm-1`:** The algorithm was very stable. The initial segmentation at 1 iteration was already accurate, and further iterations only made minor refinements.
* **For `asm-2`:** The process was unstable. The 3-iteration result was **worse**, as it incorrectly included a large part of the background building. By 5 iterations, the algorithm corrected this mistake and produced the cleanest and most accurate mask.

**Conclusion:** More iterations are not always better, but for complex images, they can be necessary for the algorithm to converge on a correct solution and fix initial errors.

---

## Task 2: Fully Convolutional Network (FCN)

### Objective

To implement and train an FCN-8s model for semantic segmentation on a subset of the PASCAL VOC dataset. The key experiment was to compare two different upsampling methods in the FCN decoder:

1.  **Transpose Convolution:** A *learnable* upsampling layer (`nn.ConvTranspose2d`).
2.  **Bilinear Interpolation:** A *fixed*, non-learnable upsampling method (`nn.Upsample`).

Both models used a pre-trained ResNet50 backbone and were trained for 30 epochs.

### Summary of Results

The model using **Bilinear Interpolation** consistently outperformed the model with Transpose Convolution, despite being a simpler method.

**Final Test Metrics**

| Metric | Transpose Convolution | Bilinear Interpolation | Winner |
| :--- | :--- | :--- | :--- |
| **Test Mean IoU** | 0.5819 | **0.6040** | Bilinear |
| **Test Pixel Accuracy**| 0.8782 | **0.8835** | Bilinear |
| **Test Loss** | 0.4533 | **0.4271** | Bilinear |
| **Model Parameters** | 23.71 Million | **23.58 Million** | Bilinear |

**Conclusion:** The simpler, non-learnable Bilinear Interpolation method was both more accurate and more efficient (fewer parameters). This suggests that the learnable Transpose Convolution layers added extra complexity that may have led to minor overfitting and was ultimately not necessary for this task.

---

## Task 3: Variational Autoencoder (VAE)

### Objective

To build and train a convolutional VAE on the MNIST dataset of handwritten digits. The goal was to learn a compressed "latent space" and compare the effects of its size on the model's performance.

Two models were trained for 50 epochs:
* **VAE-128:** Uses a latent dimension of 128.
* **VAE-256:** Uses a latent dimension of 256.

Performance was judged on image reconstruction quality, the quality of newly generated digits, and the smoothness of latent space interpolation.

### Summary of Results

The VAE with a **latent dimension of 128** performed slightly better in the final quantitative metrics.

**Final Loss Metrics (after 50 Epochs)**

| Metric | VAE (Latent Dim 128) | VAE (Latent Dim 256) | Winner |
| :--- | :--- | :--- | :--- |
| **Final Test Loss** | **52.4180** | 53.4342 | VAE-128 |
| **Final Recon. Loss** | **24.5858** | 25.2377 | VAE-128 |
| **Final KL Loss** | 26.3241 | 26.9918 | (N/A) |

* **Quantitative:** The VAE-128 model achieved a lower (better) test loss and reconstruction loss, meaning it was slightly more accurate at recreating the original digits.
* **Qualitative:** Visually, both models produced excellent, sharp reconstructions and very smooth, logical interpolations between digits (e.g., transforming a '7' into a '2').

**Conclusion:** A larger latent space is not always better. For the MNIST dataset, a 128-dimension space was large enough to capture all the important features efficiently. The 256-dimension space was slightly less efficient, resulting in a minor drop in reconstruction quality.