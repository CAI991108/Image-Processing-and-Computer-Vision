
# Image Processing and Computer Vision -- Python

This repository contains the code and implementation details for the 
**Image Processing and Computer Vision** consisting:

1. **Image Affine Transformation**
2. **Project 3D Points to Retina Plane**
3. **PortraitNet Segmentation Model**
4. **Gaussian Filter**
5. **Local Histogram Equalization**
6. **Disparity Estimation**

## Repository Structure

```
.
├── Image Affine Transformation/ 
│   ├── affine_transform.py    
├── Project 3D Points to Retina Plane/ 
│   └── perspective_projection.ipynb  
├── PortritNet Segmentation Model/ 
├── Gaussian Filter/
│   └── gaussian_filter.py 
├── Local Histogram Equalization/
│   └── local_hist_eq.py
├── Disparity Estimation/
├── reports/
└── README.md           # This file
```

### Image Affine Transformation

This task involves implementing an affine transformation on an image. 
The affine transformation includes scaling, rotation, and translation. 
The transformation is applied to the image using both a custom implementation and 
OpenCV's `cv2.getRotationMatrix2D()` function.

- **Affine Transformation Matrix**: The transformation matrix is constructed 
using scaling factors, rotation angle, and translation distances.
- **OpenCV Implementation**: The `cv2.getRotationMatrix2D()` function is used for comparison.

### Project 3D Points to Retina Plane

This task involves projecting 3D points onto a 2D image plane using the pinhole camera model. 
The camera's intrinsic and extrinsic parameters are used to perform the perspective projection.

- **Camera Intrinsic Matrix**: Generated using focal lengths and principal point.
- **Perspective Projection**: The 3D points are projected onto the 2D image plane 
using the camera's intrinsic and extrinsic parameters.

### PortraitNet Segmentation Model

PortraitNet is a real-time segmentation model based on a lightweight U-shape architecture. 
The model is trained on the EG1800 dataset and evaluated using the Intersection over Union (IoU) metric.

- **Encoder-Decoder Architecture**: The encoder uses MobileNet-v2 as the backbone network, 
and the decoder consists of refined residual blocks and up-sampling layers.
- **Loss Function**: Focal Loss is used to address class imbalance issues.
- **Evaluation Metrics**: IoU is used to evaluate the model's segmentation accuracy.


**Requirements**

- Python 3.x, OpenCV, NumPy, PyTorch, TensorBoard (for training visualization)

Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Gaussian Filter

This task involves implementing a Gaussian filter to blur an image. 
The Gaussian filter is applied using a kernel of size \(k \times k\), 
where \(k\) is the kernel size and \(\sigma\) is the standard deviation of the Gaussian distribution.

- **Gaussian Kernel**: The kernel is calculated using the Gaussian function 
and normalized by dividing by the sum of all elements in the kernel.
- **Edge Handling**: The image is padded with zeros before applying the filter to handle edge pixels.

### Local Histogram Equalization

This task involves implementing local histogram equalization to enhance the contrast of an image. 
Local histogram equalization is particularly effective for images with varying lighting conditions.

- **Histogram Equalization**: The contrast of the image is adjusted locally by
equalizing the histogram of small regions in the image.
- **Limitations**: Local histogram equalization may over-enhance noise in low-contrast regions 
and has higher computational complexity compared to global histogram equalization.

### Disparity Estimation

This task involves estimating the disparity map from stereo image pairs using both a block matching algorithm 
and a Siamese Neural Network.

- **Block Matching Algorithm**: The Sum of Absolute Differences (SAD) is used as a similarity measure 
to compute the disparity map.
- **Siamese Neural Network**: A neural network is trained to estimate the disparity map, 
providing smoother and less noisy results compared to the block matching algorithm.

**Requirements**

- Python 3.x , OpenCV, NumPy, PyTorch (for Siamese Neural Network), Matplotlib (for visualization)

Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## References

- [PortraitNet GitHub Repository](https://github.com/dong-x16/PortraitNet)
- [OpenCV Documentation](https://docs.opencv.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
