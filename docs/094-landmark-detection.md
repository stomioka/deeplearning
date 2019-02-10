# Landmark Detection

## Adding landmark (red dots)
We can add landmark on the images. A landmark is expressed as $(l_{nx}, l_{ny})$
![](images/094-landmark-detection-6103e80a.png)

Then we can pass the input to convnets as shown in below figure.
![](images/094-landmark-detection-e61fa380.png)

This is the basic building block for
- detecting the emotion


Note that landmark has to be consistent. So, if landmark 1 is used the left corner of the left eye on the first image, all other images uses landmark 1 for the same location of the face.
