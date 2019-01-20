# Size of the dev and test sets

## Old way of splitting data

* 70% Training
* 30% Test

or

* 60% training
* 20% development for iternations
* 20% test

These splits were reasonable when dataset is relatively small (100-10,000 samples)

If you have 1,000,000 samples, then you could use

* 98 % Training
* 2 % dev
* 2 % test

## Size of test set
* Set your test set to be big enough to give high confidence in teh overall performance of your system

Depends on applications, you may not need test set

More discussion [here](https://stomioka.github.io/deeplearning/docs/027-train-dev-test-sets.html)
