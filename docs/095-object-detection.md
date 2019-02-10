# Object detection

## Examples
Let's say, we want to build an object detection classifier.

![](images/095-object-detection-a0b2a4e8.png)

### Step 1
First, we need to prepare a training set

![](images/095-object-detection-5850f288.png)

### Step 2
Build a conv net and train it to detect if there is an object (car in this case)

![](images/095-object-detection-13a6f3bd.png)

### Step 3 Slideing with a smaller window
![](images/095-object-detection-f1b3996e.png)

### Step 4 Sliding with a larger window
Repeat woth a larger window throughout the image.
![](images/095-object-detection-eca669dc.png)

Repeat with even larger window.

![](images/095-object-detection-64a1624f.png)
