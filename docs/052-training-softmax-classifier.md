# Training a softmax classifier

## Understanding Softmax

Let

$z^{[l]}=\begin{bmatrix}
5\\
2\\
-1\\
3
\end{bmatrix}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$t=\begin{bmatrix}
e^5\\
e^2\\
e^{-1}\\
e^3
\end{bmatrix}$

then

$a^{[L]}=g^{[L]}(z^{[L]})=\begin{bmatrix}
\frac{e^5}{(e^5+e^2+e^{-1}+e^3)}\\
\frac{e^2}{(e^5+e^2+e^{-1}+e^3)}\\
\frac{e^{-1}}{(e^5+e^2+e^{-1}+e^3)}\\
\frac{e^3}{(e^5+e^2+e^{-1}+e^3)}
\end{bmatrix}=\begin{bmatrix}
frac{148.4}{176.3}\\
frac{7.4}{176.3}\\
frac{0.4}{176.3}\\
frac{20.1}{176.3}
\end{bmatrix}=\begin{bmatrix}
0.842\\
0.042\\
0.002\\
0.114
\end{bmatrix}$


Softmax regression generalizes logistic regression to C classes. If C=2 then, softmax regression reduces to logistic regression.

## How would train a neural network with a softmax output layer

### Loss function

Let

$y=\begin{bmatrix}
0\\
1\\
0\\
0
\end{bmatrix}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$a^{[L]}=\hat{y}=\begin{bmatrix}
0.3\\
0.2\\
0.1\\
0.4
\end{bmatrix}$

This example is not good one since the probability of y=1 is 0.2.

$\mathcal{L}(\hat{y},y)=-\sum_{j=1}^4y_jlog\hat{y}_j$

With given example, $y_1=y_3=y_4=0$ and $y_2=1$, so $\mathcal{L}(\hat{y},y)=-y_2log\hat{y}_2=log\hat{y}_2$ and $\hat{y}$ needs to be as big as possible to make $\mathcal{L}(\hat{y},y)$ small. This is reasonable as $y_2$ has to be close to 1.
This is a form of **maximum likelyhood estimation**.

### Cost function

$\mathcal{J}(w^{[1]},b^{[1]}), \cdots)=\frac{1}{m}\sum_{i=1}^m\mathcal{L}(\hat{y},y)$
