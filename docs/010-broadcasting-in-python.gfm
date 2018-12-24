# Broadcasting in python
<!-- TOC -->

- [Broadcasting in python](#broadcasting-in-python)
  - [Example 1](#example-1)
  - [Example 2](#example-2)
  - [Example 3](#example-3)
  - [Example 4](#example-4)
  - [General Principle](#general-principle)

<!-- /TOC -->
## Example 1

Carolies from Carb, Proteins, Fats in 100g of different foods:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Apples&nbsp;&nbsp;&nbsp;&nbsp;Beef&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Eggs&nbsp;&nbsp;&nbsp;&nbsp;Potatoes

$\begin{matrix}\text{Carb}\\\text{Protein}\\\text{Fat}\end{matrix}
\begin{bmatrix} 56.0 & 0.0 & 4.4 & 68.0 \\
                1.2 & 104.0 & 52.0 & 8.0 \\
                1.8 & 135.0 & 99.0 &0.9
\end{bmatrix}$

* Calculate % of calories for carb, protein, and fat for each product without explicit for-loop?

$\begin{matrix}\text{Carb}\\\text{Protein}\\\text{Fat}\end{matrix}
\begin{bmatrix} 56.0 & 0.0 & 4.4 & 68.0 \\
                1.2 & 104.0 & 52.0 & 8.0 \\
                1.8 & 135.0 & 99.0 &0.9
\end{bmatrix}=A$ where A is $\mathbb{R}^{3\times 4}$

```python
> import numpy as np
> A=np.array([[56.0,0.0,4.4,68.0],
>           [1.2,104.0,52.0,8.0],
>           [1.8,135.0,99.0,0.9]])
> print(A)
```
Prints out
```
[[ 56.    0.    4.4  68. ]
 [  1.2 104.   52.    8. ]
 [  1.8 135.   99.    0.9]]
```
Calculate sum for each column
```Python
> cal=A.sum(axis=0) #axis=0 vertically
> print(cal)
```
Prints out
```
[ 59.  239.  155.4  76.9]
```
Then calculate percentages
```Python
> percentage=100*A/cal.reshape(1,4)
> print(percentage)
```
Prints out
```
[[94.91525424  0.          2.83140283 88.42652796]
 [ 2.03389831 43.51464435 33.46203346 10.40312094]
 [ 3.05084746 56.48535565 63.70656371  1.17035111]]
 ```
`cal.reshape(1,4)` is an example of python broadcasting, where where you take a matrix A. So this is a `(3,4)` matrix and you divide it by a `(1,4)` matrix. And technically, after this first line of codes `cal`, the variable `cal`, is already a `(1,4)` matrix. So technically you don't need to call reshape here again

## Example 2
$\begin{bmatrix}1\\2\\3\\4\end{bmatrix} + 100$
Python will auto expand 100 to a `(1,4)` matrix of 100.

$\begin{bmatrix}1\\2\\3\\4\end{bmatrix} + \begin{bmatrix}100\\100\\100\\100\end{bmatrix} =
\begin{bmatrix}101\\102\\103\\104\end{bmatrix}$

## Example 3
$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix} +
\begin{bmatrix}100&200&300\end{bmatrix}$

$(m,n)$   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python will make $(1,n) \to (m,n)$

$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix} +
\begin{bmatrix}100&200&300\\
100&200&300
\end{bmatrix}=\begin{bmatrix}101&202&303\\
104&205&306
\end{bmatrix}$


## Example 4
$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix} +
\begin{bmatrix}100\\200\end{bmatrix}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(m,n)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Python will make $(m,1) \to (m,n)$

$\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix} +
\begin{bmatrix}100&100&100\\200&200&200\end{bmatrix}=\begin{bmatrix}101&202&303\\
104&205&306
\end{bmatrix}$

## General Principle

$(m,n)$ &nbsp;&nbsp;&nbsp; $+-*/$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(1,n)$ $\to (m,n)$

$(m,n)$ &nbsp;&nbsp;&nbsp; $+-*/$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$(m,1)$ $\to (m,n)$

$(m,1)$ &nbsp;&nbsp;&nbsp; $+-*/$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathbb{R}$ $\to (m,1)$

$\begin{bmatrix}1\\2\\3\end{bmatrix}+100=\begin{bmatrix}101\\102\\103\end{bmatrix}$
