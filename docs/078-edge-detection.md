# Edge Detection Example

The early layers of the neural network might detect edges and then the some later layers might detect cause of objects and then even later layers may detect cause of complete objects like people's faces in this case.
![](images/078-edge-detection-498c2284.png)

Vertical edges detector can be used to detect vertical lines, and you could use a horizontal detector to detect horizontal lines in the image on the left.
![](images/078-edge-detection-54e28048.png)

## How do you detect edges like above?
* Example using 6x6x1 matrix<br>
![](images/078-edge-detection-321a1742.png)

1. add a filter (or kernel) (3x3x1)
![](images/078-edge-detection-ef329364.png)

2. Convolution operation
![](images/078-edge-detection-619b6c27.png)
2.1 first cell<br>
![](images/078-edge-detection-fad205d5.png) $\Rightarrow -5$<br>
$(3\times1) + (1\times1)+(2\times1)+(0\times0)+(5\times0)+(7\times0)+(1\times-1)+(8\times -1)+(2\times-1)=-5$
![](images/078-edge-detection-2ba6ca6f.png)
2.2 second cell<br>
![](images/078-edge-detection-f83d6681.png)
![](images/078-edge-detection-aeb5b6ca.png)  $\Rightarrow -4$<br>
![](images/078-edge-detection-fbcccfdc.png)

2.3 repeat this for all cells
![](images/078-edge-detection-dec7872d.png)

## Python functions:
* In python, we use `conv-forward`
* In tensorflow: `tf.nn.conv2d`
* In keras: `conv2d`


## How vertical edge detector work?

![](images/078-edge-detection-7a2b13d3.png)
A vertical edge is a three by three region since we are using a 3 by 3 filter where there are bright pixels on the left, you do not care that much what is in the middle and dark pixels on the right. The middle in this 6 by 6 image is really where there could be bright pixels on the left and darker pixels on the right and that is why it thinks its a vertical edge over there.

##  Positive and negative edges
These filters make a difference between the light to dark versus the dark to light edges
![](images/078-edge-detection-5fcc1774.png)

## Vertical and Horizontal Edge Detection
![](images/078-edge-detection-2c943370.png)
![](images/078-edge-detection-5bb76c65.png)

Different filters allow you to find vertical and horizontal edges.

## Lerning to detect edges
* Vertical edge detections
$\begin{bmatrix}
1&0&-1\\
1&0&-1\\
1&0&-1\\
\end{bmatrix}$
* Sobel filter
$\begin{bmatrix}
1&0&-1\\
2&0&-2\\
1&0&-1\\
\end{bmatrix}$

* Scharr filter
$\begin{bmatrix}
3&0&-3\\
10&0&-10\\
3&0&-3\\
\end{bmatrix}$

* learn filter
![](images/078-edge-detection-ffc3eca3.png)
To detect edges in some complicated image, you can just learn the nine numbers of thes matrix as parameters, which you can then learn using back propagation. And the goal is to learn nine parameters so that when you take the 6x6 image, and convolve it with your 3x3 filter, that this gives you a good edge detector.

The idea you can treat these nine numbers as parameters to be learned has been one of the most powerful ideas in computer vision.
