# Logistic Regression Cost function

<!-- TOC -->

- [Logistic Regression Cost function](#logistic-regression-cost-function)

<!-- /TOC -->

$$\hat{y}=\sigma(w^Tx+b)$$
where
$$\sigma(z)=\frac{1}{1+e^{-z}}$$

and we want to interprete

$$\hat{y}=P(y=1|x)$$

in another word,


$\begin{matrix}
\text{If }y=1\text{: }P(y|x)=\hat{y}\\
\text{If }y=0\text{: }P(y|x)=1-\hat{y}\\\end{matrix}\Bigg\}\begin{matrix}
P(y|x)\end{matrix}$

is summarized by

$$P(y|x)=\hat{y}^y(1-\hat{y})^{(1-y)}$$



If $y=1$: $\hat{y}^y = \hat{y}$ and $1-y=0$, so  $\hat{y}^y(1-\hat{y})^{(1-y)}=\hat{y}$

If $y=0$: $\hat{y}^y = 1$ and $1-y=1$, so  $\hat{y}^y(1-\hat{y})^{(1-y)}=1-\hat{y}$

$\log P(y|x)= \log \hat{y}^y(1-\hat{y})^{(1-y)}$


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=y\log\hat{y}+(1-y) \log{(1-\hat{y})}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$=-L(\hat{y},y)$

 So minimizing the loss corresponds to maximizing the log of the probability.

 # Cost on *m* examples

 $P(\text{labels in training set}) = \Pi_{i=1}^mP(y^{(i)}|x^{(i)})$

 $\log P(\text{labels in training set}) = \log \Pi_{i=1}^mP(y^{(i)}|x^{(i)})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$= \Sigma_{i-1}^m \log P(y^{(i)}|x^{(i)})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$=-\Sigma_{i-1}^m L(\hat{y}^{(i)},y^{(i)})$

since

$\log P(y^{(i)}|x^{(i)}) = -L(\hat{y}^{(i)},y^{(i)})$ :arrow_left: Maximum liklihood estimation which just means to choose the parameters that maximizes this thing

Cost: $J(w,b)= \frac{1}{m}\Sigma_{i-1}^m L(\hat{y}^{(i)},y^{(i)})$

We now want to minimize the cost instead of maximizing likelihood, we've got to rid of the minus sign.
