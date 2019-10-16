# Train, validation and test sets

## Naive forcasting
take the last value and assume that the next value will be the same one, and this is called naive forecasting.
![](images/03-train-valid-test-95c2a8ed.png)

## Fixed Partitioning
![](images/03-train-valid-test-df7c8367.png)
## Roll forward Partitioning
 We start with a short training period, and we gradually increase it, say by one day at a time, or by one week at a time. At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period.
![](images/03-train-valid-test-6c5378af.png)
