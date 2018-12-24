# Logistic Regression Gradient Descent
<!-- TOC -->

- [Logistic Regression Gradient Descent](#logistic-regression-gradient-descent)

<!-- /TOC -->
recap:

:ballot_box_with_check: $z=w^Tx+b$

:ballot_box_with_check: $\hat{y}=a=\sigma{(z)}$ - prediction

:ballot_box_with_check: $L(a,y)=-(y\log(a)+(1-y)log(1-a))$ - loss of one example

![](images/lr-computation.svg)

In logistic regression, we want to modify the parameters, W and B, in order to reduce this loss.

 1. Compute derivative of loss with respect to the prediction.
 $$\frac{dL(a,y)}{da} = -\frac{y}{a}+\frac{1-y}{1-a}$$
 in Python we denote it as $da$.

 2.
 $$dz=\frac{dL}{dz}=\frac{dL(a,y)}{dz}= a-y$$
 $$= \frac{dL}{da}*\frac{da}{dz}$$
 where
 $$\frac{da}{dz}=a(1-a)$$

3.
$$\frac{dL}{dw_1}="dw_1"=x_1*dz$$
4.
$$\frac{dL}{dw_2}="dw_2"=x_2*dz$$
5.
$$\frac{dL}{db}="db"=dz=a-y$$

 If you want to do *gradient descent* with respect to just this one example, what you would do is the following;
 * $w_1:w_1-\alpha dw_1$
 * $w_2:w_2-\alpha dw_2$
 * $b:b -   \alpha db$
