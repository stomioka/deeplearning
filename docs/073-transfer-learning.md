# Transfer learning

One of the most powerful ideas in deep learning is that sometimes you can take knowledge the neural network has learned from one task and apply that knowledge to a separate task. So for example, maybe you could have the neural network learn to recognize objects like cats and then use that knowledge or use part of that knowledge to help you do a better job reading x-ray scans. This is called transfer learning.

## Image recognition
![](images/073-transfer-learning-93f80047.png)

* The NN was trained on cat pictures.
  - x= cat images
  - y= cat recognizer
* To transfer this for radiology diagnosis, you could simply swap the data.
  - x=radiology image,
  - y=diagnosis,
  - Then initialize the last output layer with new random weight parameters, and re-train on new data.

Sometimes the initial training on cat images are called "pre-training"

## Speech recognition
* Pretraining
    - x= audio clip
    - y= transcript

* A new task is to build wakeword.
  - You might take out the last layer of the neural network again and create a new output node
  - Create not just a single new output, but actually create several new layers to your neural network to try to put the labels Y for your wake word detection problem.
  - Depending on how much data you have, you might just retrain the new layers of the network or maybe you could retrain even more layers of this neural network.

---

##When does transfer learning make sense?
Transfer learning makes sense when you have a lot of data for the problem you're transferring from and usually relatively less data for the problem you're transferring to.

**The first example:**

  if there are **1,000,000 cat images** ($X_{pre-training}$), and **100 radiology images** ($X_{new\_task}$).

**The Second example:**

if there are **10,000 hr of speech recognition** ($X_{pre-training}$), and **1 hr wakeword** ($X_{new\_task}$).

### Principal

#### Transfer from A to B
  * Task A and B have the same input x. (first example had images as input)
  * $m_{\text{Task A}}\ggg m_{\text{Task B}}$ .
  * Low level features from A could be helpful for learning B.
