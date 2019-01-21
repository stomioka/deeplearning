# Avoidable bias

We talked about how you want your learning algorithm to do well on the training set but sometimes you don't actually want to do too well and knowing what human level performance is, can tell you exactly how well but not too well you want your algorithm to do on the training set.

**Cat classification 1**
--
**Reduce bias ** $\begin{cases}
\text{Humans error: 1%} \\
\text{Training error: 8%} \\
\text{Dev error 10%}
\end{cases}$


### In this case, the traing set needs to perform better, and focus on **reducing bias**.
* **Train a bigger neural networks**
* **Run training set longer.**
* **Increase the model size** ​(such as number of neurons/layers): This technique reduces bias, since it should allow you to fit the training set better. If you find that this increases variance, then use regularization, which will usually eliminate the increase in variance.
* **Modify input features based on insights from error analysis​:** Say your error analysis inspires you to create additional features that help the algorithm eliminate a particular category of errors. (We discuss this further in the next chapter.) These new features could help with both bias and variance. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance.
* **Reduce or eliminate regularization**​ (L2 regularization, L1 regularization, dropout): This will reduce avoidable bias, but increase variance.
* **Modify model architecture**​ (such as neural network architecture) so that it is more suitable for your problem: This technique can affect both bias and variance.
One method that is not helpful:
Add more training data​: This technique helps with variance problems, but it usually has no significant effect on bias.

[More on how to reduce bias and variance](https://stomioka.github.io/deeplearning/docs/029-basic-recipe-ml.html)

**Cat classification 2**
---
**reduce variance** $\begin{cases}
\text{Humans error: 7.5%} \\
\text{Training error: 8%} \\
\text{Dev error 10%}
\end{cases}$

**Think of human level error as a proxy or as a estimate for Bayes error or for Bayes optimal error.**

  By definition, human level error is worse than Bayes error because nothing could be better than Bayes error but human level error might not be too far from Bayes error.

The difference between **Bayes error or approximation of Bayes error** and the **training error** to be the **avoidable bias**.

###Techniques for reducing variance
* **Add more training data​:** This is the simplest and most reliable way to address variance, so long as you have access to significantly more data and enough computational power to process the data.
* **Add regularization​ (L2 regularization, L1 regularization, dropout):** This technique reduces variance but increases bias.  • Add early stopping​ (i.e., stop gradient descent early, based on dev set error): This technique reduces variance but increases bias. Early stopping behaves a lot like regularization methods, and some authors call it a regularization technique.
* **Feature selection to decrease number/type of input features:**​ This technique might help with variance problems, but it might also increase bias. Reducing the number of features slightly (say going from 1,000 features to 900) is unlikely to have a huge effect on bias. Reducing it significantly (say going from 1,000 features to 100—a 10x reduction) is more likely to have a significant effect, so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful.
* **Decrease the model size ​(such as number of neurons/layers):** ​Use with caution.
​ This technique could decrease variance, while possibly increasing bias. However, typically this is not recommend technique for addressing variance. Adding regularization usually gives better classification performance. The advantage of reducing the model size is reducing your computational cost and thus speeding up how quickly you can train models. If speeding up model training is useful, then by all means consider decreasing the model size. But if your goal is to reduce variance, and you are not concerned about the computational cost, consider adding regularization instead.
* **Modify input features based on insights from error analysis​:** Say your error analysis inspires you to create additional features that help the algorithm to eliminate a particular category of errors. These new features could help with both bias and variance. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance.
* Modify model architecture​ (such as neural network architecture) so that it is more suitable for your problem: This technique can affect both bias and variance.

## Summary
**Avoidable bias** is difference between two of these

$\text{Humans error: 7.5%} \\
\text{Training error: 8%}$

**Variance** is difference between

$\text{Training error: 8%} \\
\text{Dev error 10%}$
