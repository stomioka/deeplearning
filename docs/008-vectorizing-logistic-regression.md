# Vectorizing Logistic regression
<!-- TOC -->

- [Vectorizing Logistic regression](#vectorizing-logistic-regression)
  - [First training example](#first-training-example)
  - [Second training example](#second-training-example)
  - [Third training example](#third-training-example)
  - [Vectorizing approach](#vectorizing-approach)
    - [Compute $Z$](#compute-z)
    - [Compute $A$](#compute-a)

<!-- /TOC -->
How you can vectorize the implementation of logistic regression, so they can process an entire training set, that is implement a single elevation of grading descent with respect to an entire training set <u>without using even a single explicit for loop</u>?

If you have _m_ examples, then to make a prediction on the first example, you need to compute followings:

## First training example
$z^{(1)}=wTx^{(1)}+b$

$a^{(1)}=\sigma(z^{(1)})$

## Second training example
$z^{(2)}=wTx^{(2)}+b$

$a^{(2)}=\sigma(z^{(2)})$

## Third training example
$z^{(3)}=wTx^{(3)}+b$

$a^{(3)}=\sigma(z^{(3)})$

repeat for all _m_ traing examples.

## Vectorizing approach

### Compute $Z$

When you stack the lower case x's corresponding to a different training examples, horizontally you get a variable X.

$X=\begin{bmatrix}
|       &     |   &        &    |   \\
x^{(1)} & x^{(2)} & \ldots & x^{(m)}\\
|       &     |   &        &    |   \\
\end{bmatrix}$,



&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:arrow_up:: This is $(n_x, m)$ matrix $\mathbb{R}^{n_x\times m}$

Turns out,

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:arrow_down:: This is a $\mathbb{R}^{1\times m}$ vector

⭐ $Z=\begin{bmatrix}
z^{(1)} & z^{(2)} & \cdots & z^{(m)}
\end{bmatrix} = w^TX+\begin{bmatrix}
b & b & \cdots & b
\end{bmatrix}$ ⭐

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:arrow_up:: This is a raw vector like $\begin{bmatrix}\cdots & \cdots & \cdots
\end{bmatrix}$.

In order to implement $Z=w^TX+\begin{bmatrix}
b & b & \cdots & b
\end{bmatrix}$ in python you write
```Python
Z=np.dot(w.T, x)+b
```
In python, $b$ is a raw number, but if you add this to a matrix vector, python automatically add it up to each element in the matrix. This is called **'broaccasting'**.

$=\begin{bmatrix}w^Tx^{(1)}+b & w^Tx^{(2)}+b & \cdots & w^Tx^{(m)}+b\end{bmatrix}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:arrow_up:: This is also a $\mathbb{R}^{1\times m}$ vector


### Compute $A$
⭐A= $\begin{bmatrix}
a^{(1)} & a^{(2)} & \cdots & a^{(m)}
\end{bmatrix} =\sigma(Z)$ ⭐
