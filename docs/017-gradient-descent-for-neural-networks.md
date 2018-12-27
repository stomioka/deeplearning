# Gradient Descent for Neural networks

## Parameters
Parameters for a single layer neural network: $w^{[1]}$, $b^{[1]}$,  $w^{[2]}$, $b^{[2]}$

If
* $n_x=n^{[0]}$ input features ,
* $n^{[1]}$ Hidden units,
* $n^{[2]}=1$ Output units$:

* Matrix of $w^{[1]}$ is $(n^{[1]}, n^{[0]})$,
* Matrix of $b^{[1]}$ is $(n^{[1]}, 1)$,
* Matrix of $w^{[2]}$ is $(n^{[2]}, n^{[1]})$,
* Matrix of $b^{[2]}$ is $(n^{[2]}, 1)$,

## Cost function
$J(w^{[1]}, b^{[1]},  w^{[2]}, b^{[2]})=\frac{1}{m}L(\hat{y},y)$

where $\hat{y}=a^{[2]}$

## Gradient descent
 To train the algorithm, you need to compute gradient descent. When you are training a neural network it is important to initialize the parameters randomly rounded into all zeros.


Repeat
$\begin{cases}
\text{Compute predictions: }(\hat{y}^{(i)}, i=1, \cdots ,m)\\
\text{Compute derivative: }dw^{[1]}=\frac{dJ}{dw^{[1]}}, db^{[1]}=\frac{dJ}{db^{[1]}}, dw^{[2]}=\frac{dJ}{dw^{[2]}}, db^{[2]}=\frac{dJ}{db^{[2]}}, \\
\text{Update parameter: }w^{[1]}=w^{[1]} -\alpha{w^{[1]}}\\
\text{Update parameter: }b^{[1]}=b^{[1]} -\alpha{b^{[1]}}\\
\text{Update parameter: }w^{[2]}=w^{[2]} -\alpha{w^{[2]}}\\
\text{Update parameter: }b^{[2]}=b^{[2]} -\alpha{b^{[2]}}
\end{cases}$

## Formula for computing Derivatives

### Forward propagation:

$Z^{[1]}=W^{[1]}X+b^{[1]}$

$A^{[1]}=g^{[1]}(Z^{[1]})$

$Z^{[2]}=W^{[2]}A^{[1]}+b^{[2]}$

$A^{[2]}=g^{[2]}(Z^{[2]})$

### Backpropagation (Compute Derivatives):

$dZ^{[2]}=A^{[2]}-Y$ where $Y=[y^{(1)},y^{(2)},\cdots, y^{(m)}] \tag1$

$dw^{[2]}=\frac{1}{m}dZ^{[1]}A^{[1]T} \tag2$

$db^{[2]}=$ `(1/m)*np.sum(dZ^{[2]}, axis=1, keepdims=True)` $\tag3$

$\underbrace{\text{(1/m)*np.sum(dZ^{[2]}, axis=1, keepdims=True)} }_{\text{keepdims option will ensure that the output is a matrix of }{(n^{[2]},1)}}$

$dZ^{[1]}=\underbrace{W^{[2]T}dZ^{[2]}}_{{(n^{[1]}, m})}\underbrace{*}_{\text{element wise product}}\underbrace{g^{[1]}(z^{[1]})}_{({m^{[1]},m}) \tag4}$

$dW^{[1]}=\frac{1}{m}dZ^{[1]}X^T \tag5$

$db^{[1]}=$ `(1/m)*np.sum(dZ^{[1]}, axis=1, keepdims=True)` $\tag6$
