# Avoidable bias

We talked about how you want your learning algorithm to do well on the training set but sometimes you don't actually want to do too well and knowing what human level performance is, can tell you exactly how well but not too well you want your algorithm to do on the training set.

**Cat classification 1**
--
**Reduce bias ** $\begin{cases}
\text{Humans error: 1%} \\
\text{Training error: 8%} \\
\text{Dev error 10%}
\end{cases}$


In this case, the traing set needs to perform better, and focus on **reducing bias**.
* train a bigger neural networks
* run training selt longer.


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

**Avoidable bias** is difference between two of these

$\text{Humans error: 7.5%} \\
\text{Training error: 8%}$

**Variance** is difference between

$\text{Training error: 8%} \\
\text{Dev error 10%}$
