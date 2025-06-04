# Setup

```
# 1. Download
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz

# 2. Extract
tar xzvf images.tar.gz

# 3. Keep images for only 5 cat classes
find images/* -type f ! -regex  '\(.*Abyssinian.*\|.*Bengal.*\|.*Bombay.*\|.*Egyptian_Mau.*\|.*Russian_Blue.*\)$' -delete

# 4. Install Python packages
uv sync
```

# Usage

```
uv run main.py
```

# Observations

- The accuracy is kind of mid largely because of the architecture I believe. The layers themselves should be fine.
- Runs reasonably snappy after optimising the MaxPool2D and Convolution layers and removing the naive, raw-dogged implementations
- Accuracy caps out around 50% and I suspect this to be because the model picks up all the easy-to-learn features; I could experiment w/ a slightly deeper architecture and more diverse image preprocessing (I primarily had random rotations in mind)

# Acknowledgements

## Dataset

- [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## Convolutional Neural Networks

- [Ujjwal Karn's Blog - An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)
- [IBM Technology - What are Convolutional Neural Networks (CNNs)?](https://youtu.be/QzY57FaENXg?feature=shared)
- [DeepLearningBook - ConvNets](https://www.deeplearningbook.org/contents/convnets.html)
- [Alescontrela/Numpy-CNN](https://github.com/Alescontrela/Numpy-CNN)
- [Stanford CS231n - Lecture Slides](https://cs231n.stanford.edu/slides/2016/winter1516_lecture7.pdf)

### Convolution

- [Explained Visually - Image Kernels](https://setosa.io/ev/image-kernels/)
- [Better Explained - Convolution](https://betterexplained.com/articles/intuitive-convolution/)
- [Wikipedia - Kernel (Image Processing)](https://en.wikipedia.org/wiki/Kernel_(image_processing))

## Questions

#### Q1. How does `im2col` work?

#### Q2. What is convolution accomplishing, intuitively?

- Linked to Signal Theory - generally an engineering requirement moreso than a Computer Science one
- Fourier series comes in: claims to be able to present any periodic function as a sum of Sine waves
- Extension to Fourier series: Fourier transform -> applies to non-periodic functions too
    - These are also called basis functions, as their amalgamation leads to a new signal
    - Shifting from a 'time-domain' to a frequency domain and vice-versa is where we can derive the convolution operator
    - When we chip off higher frequencies in a frequency domain (low-pass filter), and project it back to a time-domain, it accomplishes the effect of **smoothing** the graph
    - This act of 'chopping' (low-pass/high-pass/band-pass filters) is effectively a multiplication w/ another signal
    - This **multiplication of signals in the frequency domain** is equivalent to the **convolution of said frequency graphs' projections in the time domain**
    - Refer to **Signal & Systems** by _Oppenheim, Wilski_ -- Chapter 4.4 & 4.5
