# Training with the cats vs. dogs dataset

Revisiting the famous Kaggle Dogs v Cats dataset: https://www.kaggle.com/c/dogs-vs-cats.

This was originally a challenge in building a classifier aimed at the world's best Machine Learning and AI Practitioners, but the technology has advanced so quickly, you'll see how you can do it in just a few minutes with some simple Convolutional Neural Network programming.

## Labeling Job
TF allows easy labeling for iamges by organizing the pictures into the folders.
![](images/02-tf-convnet-4331663d.png)

'ImageDataGenerator' in TF is the one often used to generate labels by how the pictures are organized.

## Training
![](images/02-tf-convnet-404b579a.png)

- `rescale=1./255` to normalize the data
- `flow_from_directory` to generate labels from the folders
- `targe_size` is the target image size.

## Validation
![](images/02-tf-convnet-8c32c621.png)

## Model Architecture

![](images/02-tf-convnet-1cb1e2b2.png)

## Model summary
![](images/02-tf-convnet-4c6325e7.png)

## Model fitting
![](images/02-tf-convnet-b0280c81.png)

[Notebook](python-examples/cat-vs-dog.ipynb)
