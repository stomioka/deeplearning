# Basic recipe for machine Learning

## Techniques for reducing bias
Does your algorithm have high bias? - eavluate training set performance).If yes,
1. **increase the model size** ​(such as number of neurons/layers): This technique reduces bias, since it should allow you to fit the training set better. If you find that this increases variance, then use regularization, which will usually eliminate the increase in variance.
2. **modify input features based on insights from error analysis​**: Say your error analysis inspires you to create additional features that help the algorithm eliminate a particular category of errors.  These new features could help with both bias and variance. In theory, adding more features could increase the variance; but if you find this to be the case, then use regularization, which will usually eliminate the increase in variance.
3. try traing longer
4. **Reduce or eliminate regularization​ (L2 regularization, L1 regularization, dropout)**: This will reduce avoidable bias, but increase variance.
5. Modify model architecture​ (such as neural network architecture) so that it is more suitable for your problem: This technique can affect both bias and variance.


## Error analysis on the training set

 For example, suppose you are building a speech recognition system for an app and have collected a training set of audio clips from volunteers. If your system is not doing well on the training set, you might consider listening to a set of ~100 examples that the algorithm is doing poorly on to understand the major categories of training set errors. Similar to the dev set error analysis, you can count the errors in different categories.

 |Audio clip | Loud background  noise| User spoke  quickly| Far from  microphone | Comments |
 |-----------|-----------------------|--------------------|----------------------|----------|
 |1   |  ✔  |    |   |   Car noise  |
 |2   | ✔   |    | ✔  |    Restaurant noise  |
 |3   |   |   ✔ |  ✔  |     User shouting across living room?  |
 |4   | ✔  |   |   | Coffee Shop  |
 |% of total   |75%   |25%   |50%   |   |


 In this example, you might realize that your algorithm is having a particularly hard time with training examples that have a lot of background noise. Thus, you might focus on techniques that allow it to better fit training examples with background noise.

You might also double-check whether it is possible for a person to transcribe these audio clips, given the same input audio as your learning algorithm. If there is so much background noise that it is simply impossible for anyone to make out what was said, then it might be unreasonable to expect any algorithm to correctly recognize such utterances.

## Techniques for reducint Variance
Do you have high variance? - evaluate dev set performance

1. **Add more training data​**: This is the simplest and most reliable way to address variance, so long as you have access to significantly more data and enough computational power to process the data.
2. **Add regularization**​ (L2 regularization, L1 regularization, dropout): This technique reduces variance but increases bias.
3. **Add early stopping**​ (i.e., stop gradient descent early, based on dev set error): This technique reduces variance but increases bias. Early stopping behaves a lot like regularization methods, and some authors call it a regularization technique.
4. Feature selection to decrease number/type of input features:​ This technique might help with variance problems, but it might also increase bias. Reducing the number of features slightly (say going from 1,000 features to 900) is unlikely to have a huge effect on bias. Reducing it significantly (say going from 1,000 features to 100—a 10x reduction) is more likely to have a significant effect, so long as you are not excluding too many useful features. In modern deep learning, when data is plentiful, there has been a shift away from feature selection, and we are now more likely to give all the features we have to the algorithm and let the algorithm sort out which ones to use based on the data. But when your training set is small, feature selection can be very useful.
5. Modify model architecture​ (such as neural network architecture) so that it is more suitable for your problem: This technique can affect both bias and variance.
6. Decrease the model size ​(such as number of neurons/layers): ​Use with caution.
​ This technique could decrease variance, while possibly increasing bias. However, generally this technique is not recommended for addressing variance. Adding regularization usually gives better classification performance. The advantage of reducing the model size is reducing your computational cost and thus speeding up how quickly you can train models. If speeding up model training is useful, then by all means consider decreasing the model size. But if your goal is to reduce variance, and you are not concerned about the computational cost, consider adding regularization instead.
