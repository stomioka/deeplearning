# Train / Dev / Test sets

## Applied machine learning is a highly iterative process

- number of layers
- number of hidden units
- learning rate
- activation Functions

Idea -> Code -> Experiment -> Idea -> Code ...

# Train/dev/test sets

![](images/b7b261f9.png)

Traditionally until a few years ago:
- 70% Training 30% test
- 60 % Training, 20 % hold-out, 20% Test (60/20/20)

When the records are 100, 1000, or 10000.

* For big data era, (>1,000,000 records)
    * 10,000 records for 'dev' would be enough
    * 10,000 records for 'test' would be enough
  98%/1%/1%

# mismatched train/test distributions
* In the era of modern deep learning is that more and more people train on mismatched train and test distributions

For example:

* Training set could come from cat pictures from webpages
* Dev, Test sets could come from cat pictures from users using your app.

**Rule of thumbs**: Make sure dev and test come from same distribution.
* Not having a test set might be okay with only dev set -  the goal of the test set is to give an unbiased estimate of the performance of the final network. But if we don't need that unbiased estimate, then it might be okay to not have a test set.
