## Surya Prakash 
## Departments of Computer Science and Engineering (CSE) 
## Lendi Institute of Engineering and Technology

---

# Deep Learning for Computer Vision: Image Representation, Formation, and Linear Filtering

## Learning Objectives

Upon completing these notes, students will be able to:
*   Understand the fundamental principles of image formation, including light interaction with surfaces and color perception.
*   Explain various methods for representing digital images, such as matrices and functions.
*   Describe the components and processes within an image sensing pipeline, including different sensor technologies and camera properties.
*   Differentiate between point, local, and global image operations and their applications.
*   Apply common point operations like contrast reversal and linear contrast stretching.
*   Understand the concept and application of histogram equalization for image enhancement.
*   Explain the mechanics of local operations, including moving average, box, Gaussian, and edge filters.
*   Distinguish between cross-correlation and convolution, including their mathematical properties and practical implications.
*   Recognize the importance of convolution's linearity and shift-invariance in computer vision.
*   Understand the concept of separable kernels and their computational advantages.
*   Address practical issues in linear filtering, such as filter size and boundary handling techniques.
*   Describe basic image compression techniques and quality metrics.

---

## 1. Image Formation

Images are formed when light interacts with objects and is subsequently captured by a sensor. This process involves several factors that influence the final image.

### 1.1 Factors Affecting Image Formation
The quality and characteristics of an image are determined by:
*   **Light Source Strength and Direction**: How bright the light is and from what angle it hits the object.
*   **Surface Geometry**: The 3D shape of the object.
*   **Material of the Surface**: Properties like texture, which dictate how light is reflected or absorbed.
*   **Nearby Surfaces**: Light reflected from adjacent objects can also influence the target surface.
*   **Sensor Capture Properties**: How the camera sensor records light.
*   **Image Representation and Color Space**: The chosen method for encoding color and intensity information.

### 1.2 Light Reflection Models
When light strikes a surface, several reactions are possible:

#### 1.2.1 Absorption
*   **Definition**: Some light energy is absorbed by the surface material, converting it into other forms of energy (e.g., heat).
*   **Explanation**: The amount of absorbed light depends on the material's **albedo ($\rho$)**. Surfaces with low albedo absorb more light. The absorption factor is often represented as $(1 - \rho)$.
*   **Real-world Example**: Dark-colored objects absorb more light and tend to feel warmer in sunlight than light-colored objects.

#### 1.2.2 Diffuse Reflection
*   **Definition**: Light scatters in multiple directions upon hitting a rough surface, independent of the viewing angle.
*   **Explanation**: This type of reflection makes surfaces appear equally bright from different viewing angles. **Lambert's cosine law** states that the amount of reflected light is proportional to the cosine of the angle between the surface normal and the light source.
*   **Real-world Example**: Brick, cloth, or rough wood surfaces exhibit diffuse reflection, making them visible from various angles without a strong glare.

#### 1.2.3 Specular Reflection
*   **Definition**: Light reflects off a smooth, polished surface at an angle equal to the incident angle, dependent on the viewing direction.
*   **Explanation**: This creates highlights or glare. The reflected light follows a predictable path, similar to how a mirror works.
*   **Real-world Example**: Mirrors, polished metal, or water surfaces show strong specular reflection, where the reflection's position changes with the viewer's perspective.

### 1.3 Other Light Interaction Phenomena
Beyond basic reflection, light can interact with surfaces in other ways:
*   **Transparency**: Light passes through the surface (e.g., clear glass).
*   **Refraction**: Light bends as it passes through a medium (e.g., a prism).
*   **Subsurface Scattering**: Light penetrates the surface, scatters within the material, and then exits at a different point (e.g., skin, wax).
*   **Fluorescence**: The output wavelength of light is different from the input wavelength.
*   **Phosphorescence**: Similar to fluorescence, but the emission of light continues after the excitation source is removed.

### 1.4 Bidirectional Reflectance Distribution Function (BRDF)
*   **Definition**: A mathematical function that describes how light is reflected from an opaque surface.
*   **Explanation**: BRDF models local reflection, indicating how bright a surface appears from a specific viewing direction when illuminated from another specific direction. It's crucial for realistic rendering in computer graphics.

### 1.5 Visible Light Spectrum and Color
Visible light is a small portion of the vast electromagnetic spectrum, ranging from violet to red (VIBGYOR). The color perceived by a sensor depends on the color of the light source and the color of the surface itself.

---

## 2. Image Representation

To process images computationally, they must be represented in a structured, digital format.

### 2.1 Human Eye and Color Perception
The human eye's structure dictates how we perceive color and intensity:
*   **Rods**: Responsible for detecting intensity (brightness) in low light conditions.
*   **Cones**: Responsible for capturing colors. Humans have three types of cones (S, M, L) with specific sensitivities that peak near red, green, and blue wavelengths. This is the primary reason for using the **RGB color representation**.
    *   **Interesting Fact**: The M and L cone sensitivities are stronger on the X chromosome, making males (XY) more prone to color-blindness than females (XX).
    *   **Animal Vision**: Not all animals have three cones. Night animals have 1, dogs have 2, while fish, birds, and mantis shrimp can have 4, 5, or even up to 12 different types of cones, indicating diverse color perception in nature.

### 2.2 Image as a Matrix
The simplest and most common way to represent a digital image is as a matrix.
*   **Explanation**: Each element in the matrix corresponds to a **pixel** (picture element) in the image. The value of each element represents the intensity or color information at that pixel location.
    *   **Intensity Values**: Typically, pixel values range from 0 to 255 (using 1 byte per pixel), where 0 is black and 255 is white. These values can also be normalized to a range between 0 and 1.
    *   **Color Channels**: For color images (e.g., RGB), there is a separate matrix for each color channel (Red, Green, Blue).
    *   **Size**: The size of the matrix (e.g., $M \times N$) depends on the image's resolution, which is determined by the image sensor.

**Diagram: Image as a Matrix**
```
+---+---+---+---+
| 20| 30| 40| 50|  <-- Row 1
+---+---+---+---+
| 60| 70| 80| 90|  <-- Row 2
+---+---+---+---+
|100|110|120|130|  <-- Row 3
+---+---+---+---+
  ^   ^   ^   ^
  |   |   |   |
  Col 1 2 3 4

(Example: A 3x4 grayscale image matrix, values 0-255)
```

### 2.3 Image as a Function
An image can also be conceptualized as a function, which is useful for defining operations.
*   **Continuous Function**: In the real world, an image can be thought of as a continuous function $I(x, y)$ that maps a 2D coordinate $(x, y)$ to an intensity value.
    *   Domain: $\mathbb{R}^2$ (representing spatial coordinates).
    *   Range: $\mathbb{R}$ (representing intensity, e.g., 0-255 or 0-1).
*   **Digital Image**: A digital image is a **discrete, sampled, and quantized** version of this continuous function.
    *   **Sampling**: The continuous image is measured at discrete points (pixels) on a grid, determined by the sensor's resolution.
    *   **Quantization**: The continuous intensity values are converted into a finite set of discrete levels (e.g., 256 levels for an 8-bit image), meaning values like 0.5 are not represented.

### 2.4 Color Spaces
While RGB is common, other color spaces are used for different applications:
*   **RGB (Red, Green, Blue)**: An additive color model, used for displays and cameras.
*   **CMYK (Cyan, Magenta, Yellow, Black)**: A subtractive color model, primarily used in printing as it's easier to control colors in this context.
*   **YCbCr**: Used in video and image compression (e.g., JPEG, MPEG). Y represents **luminance** (brightness), and CbCr represent **chrominance** (color information). Humans perceive luminance with higher fidelity, so it's compressed with higher quality.
*   **Other Color Spaces**: XYZ, YUV, Lab, HSV, etc., each with specific uses in various industries like printing and scanning. Standards for these are established by organizations like CIE.

### 2.5 Bayer Filter and Demosaicing
*   **Bayer Filter**: An arrangement of color filters on a camera sensor where not every sensing element captures all three RGB components.
    *   **Arrangement**: Typically, a Bayer grid has 50% green sensors, 25% red sensors, and 25% blue sensors, inspired by human visual receptors.
*   **Demosaicing Algorithms**: Since each pixel only records one color, these algorithms are used to reconstruct a full-color image by interpolating the missing color information from surrounding pixels.

**Diagram: Bayer Filter Pattern**
```
+---+---+---+---+
| G | R | G | R |
+---+---+---+---+
| B | G | B | G |
+---+---+---+---+
| G | R | G | R |
+---+---+---+---+
| B | G | B | G |
+---+---+---+---+
(Example: A 4x4 Bayer filter pattern)
```

---

## 3. Image Sensing Pipeline & Camera Technology

The process of capturing an image involves a series of steps within a camera system.

### 3.1 Image Sensing Pipeline
The general flow of image capture in a camera is:
1.  **Optics (Lens)**: Light enters through the lens.
2.  **Aperture and Shutter**: Control the amount of light and exposure time.
3.  **Sensor (CCD/CMOS)**: Light falls onto the sensor, converting photons into electrical signals.
4.  **Gain Factor**: Amplifies the signal from the sensor.
5.  **Raw Image**: The initial analog or digital signal representing the captured light.
6.  **Demosaicing**: Reconstructs full-color information from the raw sensor data (if using a Bayer filter).
7.  **Image Processing Algorithms**: Includes sharpening, white balancing, and other digital signal processing (DSP) methods to improve image quality.
8.  **Compression**: The processed image is compressed into a suitable format for storage (e.g., JPEG).

**Flowchart: Image Sensing Pipeline**
```
+-------+     +----------+     +--------+     +------+     +-----------+
| Optics| --> | Aperture/| --> | Sensor | --> | Gain | --> | Raw Image |
| (Lens)|     | Shutter  |     | (CCD/  |     |      |     |           |
+-------+     +----------+     | CMOS)  |     +------+     +-----------+
                                +--------+
                                     |
                                     v
                           +-------------+
                           | Demosaicing |
                           +-------------+
                                     |
                                     v
                           +---------------------+
                           | Image Processing    |
                           | (Sharpening, White  |
                           |  Balance, DSP)      |
                           +---------------------+
                                     |
                                     v
                           +-------------+
                           | Compression |
                           +-------------+
                                     |
                                     v
                           +---------+
                           | Storage |
                           +---------+
```

### 3.2 CCD vs. CMOS Sensors
Two primary types of image sensors are used in cameras:

*   **CCD (Charge-Coupled Device)**:
    *   **Mechanism**: Generates a charge at each sensing element (pixel) when photons strike it. These charges are then moved from pixel to pixel, typically along columns, to an output node where they are converted into a voltage. An Analog-to-Digital Converter (ADC) then converts this voltage into a digital pixel value.
    *   **Advantages**: Historically known for higher image quality and lower noise in certain applications.
    *   **Disadvantages**: Slower readout speeds, higher power consumption, and requires external ADC.

*   **CMOS (Complementary Metal-Oxide Semiconductor)**:
    *   **Mechanism**: Converts charge to voltage *within each pixel* using transistors. The signal is then amplified and moved using traditional wires. The signal is typically digital at the pixel level, often eliminating the need for an external ADC.
    *   **Advantages**: Faster readout, lower power consumption, smaller size, and integrated ADC capabilities.
    *   **Disadvantages**: Historically had more noise and lower quality, but modern CMOS technologies have largely overcome these limitations and are now dominant in most cameras, including smartphones.

### 3.3 Camera Properties
Several properties define a camera's capabilities and the characteristics of the captured image:
*   **Shutter Speed (Exposure Time)**: Controls the duration light reaches the sensor, affecting brightness and motion blur.
*   **Sampling Pitch**: The spacing between individual sensor cells on the imaging chip.
*   **Fill Factor (Active Sensing Area Size)**: The ratio of the active light-sensitive area to the total available area of a sensing element.
*   **Chip Size**: The overall physical dimensions of the sensor chip.
*   **Analog Gain (ISO)**: Amplification of the sensor signal, often controlled by the camera's ISO setting. Higher gain increases sensitivity but can also increase noise.
*   **Sensor Noise**: Unwanted variations in the signal introduced during the sensing process.
*   **Resolution**: The number of bits used to represent each pixel's intensity (e.g., 8 bits for 0-255 values).
*   **Post-processing Elements**: Digital image enhancement methods applied after raw capture but before compression.

### 3.4 DSLR vs. Mirrorless Cameras
*   **DSLR (Digital Single-Lens Reflex)**:
    *   **Mechanism**: Uses a mirror mechanism to reflect light to an optical viewfinder. The mirror flips out of the way to allow light to reach the image sensor during exposure.
    *   **Advantages**: Traditionally offered better picture quality, more functionality, physical shutter mechanisms, and variable focal length/aperture lenses.
*   **Mirrorless Cameras (e.g., Smartphones)**:
    *   **Mechanism**: Light passes directly through the lens to the sensor, with no mirror. The viewfinder (if present) is electronic.
    *   **Advantages**: More accessible, portable, and generally lower cost. Modern mirrorless cameras have significantly closed the quality gap with DSLRs.

### 3.5 Sampling and Aliasing
*   **Shannon Sampling Theorem**: States that to accurately reconstruct a continuous signal from its samples, the sampling frequency must be at least twice the maximum frequency ($f_{\text{max}}$) present in the signal. This minimum sampling rate is called the **Nyquist frequency**.
*   **Aliasing**: Occurs when the sampling rate is below the Nyquist frequency, causing higher frequencies in the original signal to be misrepresented as lower frequencies in the sampled signal.
    *   **Impact**: Aliasing can create undesirable artifacts (e.g., jagged edges, moiré patterns) when images are upsampled or downsampled.

---

## 4. Image Operations: Point, Local, and Global

Image operations transform an input image $I$ into an output image $\hat{I}$. These operations can be categorized by how each output pixel is computed.

### 4.1 Types of Operations
1.  **Point Operations**:
    *   **Definition**: The value of an output pixel $\hat{I}(x,y)$ depends *only* on the value of the corresponding input pixel $I(x,y)$ at the same coordinate.
    *   **Complexity**: Constant per pixel (O(1)).
2.  **Local Operations**:
    *   **Definition**: The value of an output pixel $\hat{I}(x,y)$ depends on the values of pixels within a *neighborhood* or region around the corresponding input pixel $I(x,y)$.
    *   **Complexity**: $p^2$ operations per pixel for a $p \times p$ neighborhood (O($p^2$)).
3.  **Global Operations**:
    *   **Definition**: The value of an output pixel $\hat{I}(x,y)$ depends on *all* pixels in the entire input image.
    *   **Complexity**: $N^2$ operations per pixel for an $N \times N$ image (O($N^2$)).

**Diagram: Types of Image Operations**
```
Input Image (I)         Output Image (I_hat)

+---+---+---+           +---+---+---+
|   |   |   |           |   |   |   |
+---+---+---+           +---+---+---+
|   | I(x,y)|           |   |I_hat(x,y)|
+---+---+---+           +---+---+---+
|   |   |   |           |   |   |   |
+---+---+---+           +---+---+---+

1. Point Operation: I_hat(x,y) depends ONLY on I(x,y)
   (e.g., brightness adjustment)

+---+---+---+           +---+---+---+
| N1| N2| N3|           |   |   |   |
+---+---+---+           +---+---+---+
| N4| I(x,y)|           |   |I_hat(x,y)|
+---+---+---+           +---+---+---+
| N5| N6| N7|           |   |   |   |
+---+---+---+           +---+---+---+

2. Local Operation: I_hat(x,y) depends on I(x,y) and its neighbors (N1-N7)
   (e.g., blurring, edge detection)

+-----------------+     +-----------------+
|                 |     |                 |
|       I         |     |     I_hat       |
|                 |     |                 |
+-----------------+     +-----------------+

3. Global Operation: I_hat(x,y) depends on ALL pixels in I
   (e.g., Fourier Transform)
```

### 4.2 Point Operations

#### 4.2.1 Contrast Reversal (Image Negative)
*   **Definition**: An image enhancement technique that inverts the intensity values, making light areas dark and dark areas light.
*   **Formula**: For a pixel at $(x,y)$, the output intensity $\hat{I}(x,y)$ is given by:
    $$\hat{I}(x,y) = I_{\text{MAX}} - I(x,y) + I_{\text{MIN}}$$
    where:
    *   $I_{\text{MAX}}$: Maximum possible intensity value (e.g., 255 for 8-bit images).
    *   $I_{\text{MIN}}$: Minimum possible intensity value (e.g., 0 for 8-bit images).
    *   $I(x,y)$: Original intensity value at pixel $(x,y)$.
    *   $\hat{I}(x,y)$: New intensity value at pixel $(x,y)$.
    If $I_{\text{MIN}} = 0$, the formula simplifies to $\hat{I}(x,y) = I_{\text{MAX}} - I(x,y)$.
*   **Example**: If $I_{\text{MAX}}=255$ and $I_{\text{MIN}}=0$:
    *   A pixel with value 240 (light gray) becomes $255 - 240 = 15$ (dark gray).
    *   A pixel with value 10 (dark gray) becomes $255 - 10 = 245$ (light gray).

#### 4.2.2 Linear Contrast Stretching
*   **Definition**: An image enhancement technique that stretches the range of intensity values in an image to utilize the full available dynamic range (e.g., 0-255), thereby increasing contrast.
*   **Explanation**: If an image's intensity values only span a narrow range (e.g., 100-200), contrast stretching remaps these values to a wider range (e.g., 0-255).
*   **Formula (Simplified Example)**:
    Let $I_{\text{min_old}}$ and $I_{\text{max_old}}$ be the minimum and maximum intensity values in the original image.
    Let $I_{\text{min_new}}$ and $I_{\text{max_new}}$ be the desired minimum and maximum intensity values (e.g., 0 and 255).
    The transformation for a pixel $I(x,y)$ is:
    $$\hat{I}(x,y) = (I(x,y) - I_{\text{min_old}}) \times \frac{I_{\text{max_new}} - I_{\text{min_new}}}{I_{\text{max_old}} - I_{\text{min_old}}} + I_{\text{min_new}}$$
    *   The ratio $\frac{I_{\text{max_new}} - I_{\text{min_new}}}{I_{\text{max_old}} - I_{\text{min_old}}}$ is the stretching factor.
*   **Example**: Assume an image has pixel values between 100 and 200. We want to stretch this to 0-255.
    *   $I_{\text{min_old}} = 100$, $I_{\text{max_old}} = 200$.
    *   $I_{\text{min_new}} = 0$, $I_{\text{max_new}} = 255$.
    *   Stretching factor = $(255-0) / (200-100) = 255 / 100 = 2.55$.
    *   A pixel with value 150 becomes: $(150 - 100) \times 2.55 + 0 = 50 \times 2.55 = 127.5 \approx 128$.
    *   This moves the middle of the old spectrum (150) to the middle of the new spectrum (128), effectively increasing contrast.

#### 4.2.3 Histogram Equalization
*   **Definition**: A more advanced contrast stretching technique that redistributes pixel intensities to flatten the image's histogram, thereby enhancing contrast, especially in images with narrow intensity ranges.
*   **Explanation**:
    1.  **Compute Histogram**: Create a histogram $h(I)$ of the image, which counts the frequency of each intensity value.
    2.  **Compute Cumulative Distribution Function (CDF)**: Integrate the histogram to get the cumulative distribution $c(I)$. This function gives the number of pixels up to a particular intensity value, normalized by the total number of pixels.
    3.  **Transform Intensities**: The new intensity value for a pixel is determined by mapping its original intensity through the normalized CDF and scaling it by the maximum intensity.
*   **Formula**:
    $$\hat{I}(x,y) = I_{\text{MAX}} \times c(I(x,y))$$
    where:
    *   $I_{\text{MAX}}$: Maximum possible intensity value (e.g., 255).
    *   $c(I(x,y))$: The normalized cumulative distribution value for the original intensity $I(x,y)$.
*   **Intuition**: If many pixels are clustered in a narrow intensity range, histogram equalization spreads them out across the full range, making subtle differences more visible.

#### 4.2.4 Noise Reduction (Point Operation)
*   **Method**: For a *still scene*, noise can be reduced by taking multiple images and averaging them pixel-wise. The random noise tends to average out over many samples.
*   **Disadvantage**: This method is impractical for dynamic scenes or when multiple images are not available.

### 4.3 Local Operations (Filtering)
Local operations are often called **filtering** and involve applying a small matrix, called a **filter**, **mask**, or **kernel**, to each pixel's neighborhood.

#### 4.3.1 Moving Average Filter (Box Filter)
*   **Definition**: A simple local operation that replaces each pixel's value with the average of its neighboring pixels within a defined window (e.g., $3 \times 3$).
*   **Purpose**: Primarily used for smoothing images and reducing noise.
*   **Mechanism**: A window (e.g., $3 \times 3$) slides across the image. At each position, the average of the pixels within the window is calculated and assigned to the central pixel of the output image.
*   **Filter Coefficients**: For a $3 \times 3$ moving average, the kernel (or filter) consists of all $1/9$ values. For a $5 \times 5$ window, it would be $1/25$.
*   **Effect**: Blurs the image, making it smoother. Can introduce "blockiness" artifacts, especially at edges, because it treats all pixels in the neighborhood equally.

**Diagram: Moving Average (Box Filter) Kernel**
```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

#### 4.3.2 Gaussian Filter
*   **Definition**: An averaging filter that uses a Gaussian function to assign weights to pixels in the neighborhood. The central pixel receives the highest weight, and weights decrease with distance from the center.
*   **Purpose**: Provides smoother blurring and noise reduction compared to the box filter, with fewer blockiness artifacts.
*   **Mechanism**: The weights in the kernel are derived from a discrete 2D Gaussian function. A normalizing factor (e.g., $1/16$) ensures pixel intensities don't blow up.
*   **Advantages**: Better at preserving edges and producing more natural-looking blur than the box filter.

**Diagram: Gaussian Filter Kernel (Example)**
```
1/16 [ 1  2  1 ]
     [ 2  4  2 ]
     [ 1  2  1 ]
```

#### 4.3.3 Edge Filters
*   **Definition**: Local operations designed to detect and highlight edges in an image.
*   **Mechanism**: These filters typically use coefficients that emphasize differences in intensity.
*   **Example: Vertical Edge Filter**:
    ```
    -1  0  1
    -1  0  1
    -1  0  1
    ```
    This filter detects vertical edges by finding significant differences between pixels on the left and right sides of the neighborhood.
*   **Example: Horizontal Edge Filter**:
    ```
    -1 -1 -1
     0  0  0
     1  1  1
    ```
    This filter detects horizontal edges by finding significant differences between pixels above and below the neighborhood.
*   **Note**: Normalization factors are often omitted in edge filters if only the presence of an edge (high intensity difference) is important, not its absolute value.

### 4.4 Global Operations
*   **Definition**: Operations where the output at any pixel depends on the entire input image.
*   **Example**: The **Fourier Transform** is a classic example of a global operation, converting an image from the spatial domain to the frequency domain.

---

## 5. Linear Filtering: Correlation vs. Convolution

Linear filtering is a fundamental concept in image processing, involving operations like cross-correlation and convolution.

### 5.1 Filter, Mask, or Kernel
These terms are used interchangeably to refer to the small matrix of coefficients applied during local operations. The entries of this matrix are called **filter coefficients**.

### 5.2 Cross-Correlation
*   **Definition**: A linear filtering operation that measures the similarity between a filter (kernel) and a local neighborhood of an image. It's essentially a dot product between the filter and the image patch.
*   **Formula**: For an input image $I$ and a filter $H$ of size $(2k+1) \times (2k+1)$, the output $G$ at location $(i,j)$ is:
    $$G(i,j) = \sum_{u=-k}^{k} \sum_{v=-k}^{k} H(u,v) I(i+u, j+v)$$
    where:
    *   $H(u,v)$: Coefficient of the filter at relative coordinates $(u,v)$.
    *   $I(i+u, j+v)$: Pixel value in the input image at coordinates $(i+u, j+v)$.
*   **Impulse Signal Response**: If cross-correlation is applied to an impulse signal (a single white pixel in a black image), the output is a *double-flipped* version of the filter. This means it does not act as an identity operation.

**Diagram: Cross-Correlation with Impulse Signal**
```
Input Impulse:         Filter H:
  0 0 0                  a b c
  0 1 0                  d e f
  0 0 0                  g h i

Cross-Correlation Output:
  i h g
  f e d
  c b a
(The filter is double-flipped/rotated 180 degrees)
```

### 5.3 Convolution
*   **Definition**: A linear filtering operation very similar to cross-correlation, but with a crucial difference: the filter (kernel) is *double-flipped* (rotated by 180 degrees) before being applied to the image.
*   **Formula**: For an input image $I$ and a filter $H$ of size $(2k+1) \times (2k+1)$, the output $G$ at location $(i,j)$ is:
    $$G(i,j) = \sum_{u=-k}^{k} \sum_{v=-k}^{k} H(u,v) I(i-u, j-v)$$
    Note the $(i-u, j-v)$ in the input image coordinates, which accounts for the filter flip.
*   **Impulse Signal Response**: When convolution is applied to an impulse signal, the output is the filter itself. This makes convolution an **identity operation** with respect to an impulse.
*   **Why Convolution?**:
    *   **Mathematical Elegance**: Convolution possesses several desirable mathematical properties that cross-correlation generally lacks.
    *   **Biological Inspiration**: Early experiments in computer vision showed that simple cells in the mammalian visual cortex perform linear spatial summation over their receptive fields, hinting that convolution could model these biological processes.

**Diagram: Convolution with Impulse Signal**
```
Input Impulse:         Filter H:
  0 0 0                  a b c
  0 1 0                  d e f
  0 0 0                  g h i

Convolution Output:
  a b c
  d e f
  g h i
(The output is the filter itself)
```

### 5.4 Properties of Convolution
Convolution is a **linear shift-invariant operator**. These properties are fundamental to its widespread use in deep learning and computer vision.

#### 5.4.1 Linearity (Superposition Principle)
*   **Definition**: Convolution is linear, meaning it satisfies both additivity and homogeneity.
*   **Additivity**: Convolving the sum of two images with a filter is the same as summing the convolutions of each image with the filter:
    $$(I_1 + I_2) * H = (I_1 * H) + (I_2 * H)$$
*   **Homogeneity**: Scaling an image before convolution is the same as scaling the result of convolution:
    $$(cI) * H = c(I * H)$$
    where $c$ is a scalar.

#### 5.4.2 Shift Invariance (Translation Invariance)
*   **Definition**: Shifting or translating the input image results in a similarly shifted output image, but the effect of the operator (filter) remains the same regardless of the object's position in the image.
*   **Importance in Computer Vision**: This property is crucial because it means a filter designed to detect a feature (e.g., a cat) will detect that feature regardless of where it appears in the image (e.g., top-left or bottom-right). This makes convolutional neural networks (CNNs) robust to object location.

#### 5.4.3 Commutativity
*   **Definition**: The order of convolution does not matter.
*   **Formula**: $I * H = H * I$
*   **Implication**: You can convolve an image with a filter, or a filter with an image, and get the same result.

#### 5.4.4 Associativity
*   **Definition**: When applying multiple convolutions, the grouping of operations does not matter.
*   **Formula**: $(I * H_1) * H_2 = I * (H_1 * H_2)$
*   **Implication**: This allows for pre-computing the convolution of multiple filters ($H_1 * H_2$) into a single filter, which can then be applied to the image, saving computational cost if the same filter combination is used repeatedly.

#### 5.4.5 Distributivity
*   **Definition**: Convolution distributes over addition.
*   **Formula**: $I * (H_1 + H_2) = (I * H_1) + (I * H_2)$

#### 5.4.6 Identity with Unit Impulse
*   **Definition**: Convolving an image with a unit impulse function (a single 1 surrounded by zeros) yields the original image itself.
*   **Formula**: $I * \delta = I$
*   **Implication**: The unit impulse acts as the identity element for convolution.

### 5.5 Separability of Kernels
*   **Definition**: A 2D kernel $K$ is **separable** if it can be expressed as the outer product of two 1D vectors (a vertical vector $v$ and a horizontal vector $h$).
    $$K = v h^T$$
*   **Computational Advantage**: For a $k \times k$ kernel, a typical 2D convolution requires $k^2$ operations per pixel. If the kernel is separable, it can be performed as two sequential 1D convolutions (first horizontal, then vertical), reducing the operations to $2k$ per pixel.
    *   **Example**: For a $5 \times 5$ filter, $k^2 = 25$ operations, but $2k = 10$ operations if separable. This is a significant speedup for larger kernels.
*   **Examples of Separable Kernels**: Box filters, Gaussian filters, and some edge filters are separable.
*   **Detecting Separability (SVD)**: The most principled way to determine if a kernel $K$ is separable is by performing its Singular Value Decomposition (SVD).
    *   **SVD**: Any matrix $K$ can be decomposed as $K = U \Sigma V^T$, where $U$ and $V$ are orthogonal matrices, and $\Sigma$ is a diagonal matrix containing singular values ($\sigma_i$).
    *   **Condition**: If $K$ has only one non-zero singular value ($\sigma_1$), then it is separable. The 1D vertical kernel $v$ is proportional to $\sqrt{\sigma_1} u_1$ (first column of $U$) and the 1D horizontal kernel $h$ is proportional to $\sqrt{\sigma_1} v_1$ (first column of $V$).

**Diagram: Separable Convolution**
```
Original 2D Kernel K (e.g., Gaussian)
  [ 1  2  1 ]
  [ 2  4  2 ]
  [ 1  2  1 ]

Can be decomposed into:

Vertical 1D Kernel v:   Horizontal 1D Kernel h:
  [ 1 ]                   [ 1  2  1 ]
  [ 2 ]
  [ 1 ]

Process:
1. Convolve image rows with h (1D horizontal convolution)
2. Convolve intermediate result columns with v (1D vertical convolution)
```

### 5.6 Practical Issues with Filters

#### 5.6.1 Ideal Filter Size
*   **Trade-offs**: The choice of filter size (e.g., $3 \times 3$, $5 \times 5$, $7 \times 7$) depends on the application.
    *   **Larger Filters**:
        *   **Advantages**: Reduce noise variance more effectively, as more neighbors contribute to the average.
        *   **Disadvantages**: Lead to more blurring, spread the impact of noise over a larger region (larger receptive field), and are computationally more expensive ($k^2$ operations).
    *   **Smaller Filters**:
        *   **Advantages**: Less blurring, less computational cost.
        *   **Disadvantages**: Less effective at noise reduction, more susceptible to local noise.

#### 5.6.2 Boundary Handling
When a filter is applied near the edges of an image, parts of the kernel extend beyond the image boundaries. This poses a problem because there are no pixel values to multiply with the filter coefficients.
*   **Problem**:
    *   **Shrinking Output**: Without special handling, the output image will be smaller than the input image. For a $100 \times 100$ image and a $3 \times 3$ filter, the output would be $98 \times 98$ pixels, losing information at the borders.
*   **Padding Strategies**: To maintain the output image size and handle boundaries, the input image can be "padded" with artificial values:
    1.  **Zero Padding**: The most common method, where pixels outside the image boundary are assumed to be zero (black).
    2.  **Wrap Around**: The image is treated as toroidal, meaning pixels from the opposite side of the image are used for padding.
    3.  **Clamp (Replicate)**: The values of the nearest edge pixels are extended outwards to fill the padded region.
    4.  **Mirror (Reflect)**: The image content is reflected across the boundaries to create the padded values.
*   **Impact**: While these strategies prevent image shrinking, they can introduce minor artifacts at the edges, though the visual impact is often minimal for larger images.

**Diagram: Boundary Padding Strategies (Conceptual)**
```
Original Image (3x3):
+---+---+---+
| A | B | C |
+---+---+---+
| D | E | F |
+---+---+---+
| G | H | I |
+---+---+---+

1. Zero Padding (with 1 pixel border):
+---+---+---+---+---+
| 0 | 0 | 0 | 0 | 0 |
+---+---+---+---+---+
| 0 | A | B | C | 0 |
+---+---+---+---+---+
| 0 | D | E | F | 0 |
+---+---+---+---+---+
| 0 | G | H | I | 0 |
+---+---+---+---+---+
| 0 | 0 | 0 | 0 | 0 |
+---+---+---+---+---+

2. Clamp/Replicate Padding:
+---+---+---+---+---+
| A | A | B | C | C |
+---+---+---+---+---+
| A | A | B | C | C |
+---+---+---+---+---+
| D | D | E | F | F |
+---+---+---+---+---+
| G | G | H | I | I |
+---+---+---+---+---+
| G | G | H | I | I |
+---+---+---+---+---+

3. Mirror/Reflect Padding:
+---+---+---+---+---+
| I | H | I | H | G |
+---+---+---+---+---+
| F | E | F | E | D |
+---+---+---+---+---+
| C | B | C | B | A |
+---+---+---+---+---+
| F | E | F | E | D |
+---+---+---+---+---+
| I | H | I | H | G |
+---+---+---+---+---+
```

---

## 6. Image Compression

After an image is captured and processed, it needs to be stored efficiently, which often involves compression.

### 6.1 YCbCr Color Space for Compression
*   **Purpose**: Images are often converted to the YCbCr color space before compression.
*   **Components**:
    *   **Y (Luminance)**: Represents the brightness or intensity information.
    *   **CbCr (Chrominance)**: Represents the color information (blue-difference and red-difference chroma components).
*   **Advantage**: The human visual system is more sensitive to changes in luminance than chrominance. Therefore, during compression, luminance (Y) can be preserved with higher fidelity, while chrominance (CbCr) can be compressed more aggressively (e.g., by subsampling) without a significant perceived loss in quality.

### 6.2 Discrete Cosine Transform (DCT)
*   **Definition**: A widely used mathematical transform in image and video compression standards like JPEG and MPEG.
*   **Mechanism**: DCT transforms spatial domain image data into the frequency domain. It concentrates most of the image information into a few low-frequency coefficients, allowing high-frequency (less perceptually important) coefficients to be discarded or quantized more heavily, leading to compression.
*   **Relationship to Fourier Transform**: DCT is a variant of the Discrete Fourier Transform (DFT) and can be seen as a reasonable approximation of an eigen decomposition of image patches.

### 6.3 Video Compression
*   **Block-Level Motion Compensation**: Videos are compressed by dividing them into frames and then into blocks within frames. Motion compensation techniques exploit temporal redundancy between frames by predicting the movement of blocks.
*   **MPEG Standard**: Divides frames into different types for efficient coding:
    *   **I-frames (Intra-coded frames)**: Independently encoded frames, similar to a JPEG image. They contain full image data and serve as reference points.
    *   **P-frames (Predicted frames)**: Encoded using motion compensation from a previous I-frame or P-frame. They store only the differences from the reference frame.
    *   **B-frames (Bi-directionally predicted frames)**: Encoded using motion compensation from both previous and future I-frames or P-frames, offering higher compression ratios.

### 6.4 PSNR (Peak Signal-to-Noise Ratio)
*   **Definition**: A common statistical metric used to quantify the quality of image compression or reconstruction.
*   **Formula**:
    $$\text{PSNR} = 10 \log_{10} \left( \frac{I_{\text{MAX}}^2}{\text{MSE}} \right)$$
    where:
    *   $I_{\text{MAX}}$: The maximum possible pixel intensity value (e.g., 255 for 8-bit images).
    *   **MSE (Mean Squared Error)**: The average of the squared differences between the original image and the compressed/reconstructed image, calculated pixel-wise.
        $$\text{MSE} = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} (I(i,j) - \hat{I}(i,j))^2$$
        where $M \times N$ is the image size, $I$ is the original image, and $\hat{I}$ is the compressed image.
*   **Interpretation**: A higher PSNR value generally indicates better image quality (less distortion from compression). While widely used, PSNR does not always perfectly correlate with human perceptual quality.

---
 **padding strategies** (zero, clamp, mirror) to prevent output image shrinkage.
*   **Image Compression** often leverages color spaces like YCbCr and transforms like DCT to reduce file size, with PSNR as a common quality metric. Video compression further uses motion compensation and different frame types (I, P, B).
