# Orthogonalization

The basic idea about orthogonalization is that you would like to implement controls that only affect a single component of your algorithms performance at a time. For example, to address bias problems you could use a bigger network or more robust optimization techniques. You would like these controls to only affect bias and not other issues such as poor generalization. An example of a control which lacks orthogonalization is stopping your optimization procedure early (early stopping). This is because it simultaneously affects the bias and variance of your model.

## Chain of assumptions in ML

- Fit training set well on cost function
  - Performance on the training set needs to pass some acceptability assessment. For some applications, this might mean doing comparably to human level performance.
  - maybe use **bigger network**
  - Use different **optimization** algorithms e.g. Adam
- Fit dev set well on cost function
  - Use different **regularization**
  - Use **big training set**
- Fit test set well on cost function
  - Use **big dev set**
- Performs well in real world
  - Go back and change **dev, test set or cost function**

Early stopping makes difficult for orthogonalization process.
