# Forward Propagation in a Deep Network

## Example of forward propagation in a 4 layer neural networks
![](images/020-forward-propagation-in-a-deep-network-1ace7177.png)

A single training example, $x$.

For layer 1:
$\begin{align}\begin{split}
x: & z^{[1]}=w^{[1]}\overbrace{x}^{\color{red}{a^{[0]}}}+b^{[1]}\\
&a^{[1]}=g^{[1]}(z^{[1]})
 \end{split}\end{align}$
For layer 2:
$\begin{align}\begin{split}
& z^{[2]}=w^{[2]}a^{[1]}+b^{[2]}\\
&a^{[2]}=g^{[2]}(z^{[2]})
\end{split}\end{align}$
For layer 3:
$\begin{align}\begin{split}
& z^{[3]}=w^{[3]}a^{[2]}+b^{[3]}\\
&a^{[3]}=g^{[3]}(z^{[3]})\\
\end{split}\end{align}$
For layer 4:
$\begin{align}\begin{split}
& z^{[3]}=w^{[3]}a^{[3]}+b^{[4]}\\
&a^{[4]}=g^{[4]}(z^{[4]})
\end{split}\end{align}$

### Non-Vectorized Forward Propagation

$z^{[l]}=w^{[l]}a^{[l-1]}+b^{[l]}$

$a^{[l]}=g^{[l]}(z^{[l]})$

## Vectorized implementation of forward Propagation
* $x^{(1)},x^{(2)}, x^{(3)} , \cdots, x^{(m)}$ are stacked togeter in columns.
$X=\begin{bmatrix}|&|&|&|\\
x^{(1)}&x^{(2)}& \cdots & x^{(m)}\end{bmatrix}$
* Similary, $z^{[1](1)},z^{[1](2)}, z^{[1](3)} , \cdots, z^{[1](m)}$ are stacked togeter in columns.
$Z^{[1]}=\begin{bmatrix}|&|&|&|\\
z^{[1](1)}&z^{[1](2)}& \cdots & z^{[1](m)}\end{bmatrix}$
* $a^{[1](1)},a^{[1](2)}, a^{[1](3)} , \cdots, a^{[1](m)}$ are stacked togeter in columns.
$A^{[1]}=\begin{bmatrix}|&|&|&|\\
a^{[1](1)}&a^{[1](2)}& \cdots & a^{[1](m)}\end{bmatrix}$
* Parameters $w$ and $b$ are done differently.

For layer 1:
$\begin{align}\begin{split}
& Z^{[1]}=w^{[1]}\overbrace{X}^{\color{red}{A^{[0]}}}+b^{[1]}\\
&A^{[1]}=g^{[1]}(Z^{[1]})
 \end{split}\end{align}$
 For layer 2:
 $\begin{align}\begin{split}
 & Z^{[2]}=w^{[2]}A^{[1]}+b^{[2]}\\
 &A^{[2]}=g^{[2]}(Z^{[2]})
  \end{split}\end{align}$
 For layer 3:
 $\begin{align}\begin{split}
 & Z^{[3]}=w^{[3]}A^{[2]}+b^{[3]}\\
 &A^{[3]}=g^{[3]}(Z^{[3]})
  \end{split}\end{align}$
 For layer 4:
 $\begin{align}\begin{split}
 & Z^{[4]}=w^{[4]}A^{[3]}+b^{[4]}\\
 & A^{[4]}=g^{[4]}(Z^{[4]})\\
 &=\hat{Y}
  \end{split}\end{align}$

  ## Vectorized Forward Propagation

  $Z^{[l]}=w^{[l]}A^{[l-1]}+b^{[l]}$

  $A^{[l]}=g^{[l]}(Z^{[l]})$

  then we have to use for loop: `for l in range (1,5,1):`
