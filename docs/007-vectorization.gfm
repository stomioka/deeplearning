#Vectorization
<!-- TOC -->

- [What is vectorization?](#what-is-vectorization)
- [Non vectorization approach](#non-vectorization-approach)
- [Vectorized approach](#vectorized-approach)
- [Experiment](#experiment)
- [Examples](#examples)
  - [Non vectorized implementation](#non-vectorized-implementation)
  - [Vectorized implementation](#vectorized-implementation)
  - [Other numpy functions](#other-numpy-functions)
  - [Logistic Regression Derivatives](#logistic-regression-derivatives)
  - [Logistic Regression Derivatives with vectorized approach](#logistic-regression-derivatives-with-vectorized-approach)

<!-- /TOC -->
## What is vectorization?
In logistic regression you need to compute $z=w^Tx+b$ where

$w=\begin{bmatrix}
    .\\[0.3em]\vdots\\[0.3em].\\[0.3em]
  \end{bmatrix} \in \mathbb {R}^{n_x}$
   and
$x=\begin{bmatrix}
.\\[0.3em]\vdots\\[0.3em].\\[0.3em]
  \end{bmatrix} \in \mathbb {R}^{n_x}$

## Non vectorization approach
Performance is much slower with for loop.
```Python
z=0
for i in range(n-x):
  z+= w[i]*x[i]
z+=b
```
## Vectorized approach
Vectorized approach is much more efficient that for loop.
```Python
z=np.dot(w,x)+b # it calculates w^Tx
```

## Experiment
Compare the time to calculate a dot product using for loop and vector.
```Python
import time

a=np.random.rand(1000000)
b=np.random.rand(1000000)

tic=time.time()
c=np.dot(a,b)
toc=time.time()

print(c)
print("vectorized version:" +str(1000*(toc-tic)) +" ms")

c=0
tic=time.time()
for i in range(100000):
    c += a[i]*b[i]
toc=time.time()

print(c)
print("for loop version:" +str(1000*(toc-tic)) +" ms")

```
249894.81121245175<br>
vectorized version:0.9729862213134766 ms<br>
249894.81121245382<br>
for loop version:327.1458148956299 ms<br>

It turns out that for loop took 327 times longer to compute.
So, whenever possible, avoid explicit for loops.

## Examples

$v=\begin{bmatrix}
    v_1\\[0.3em]\vdots\\[0.3em]v_n\\[0.3em]
  \end{bmatrix}$ $u=\begin{bmatrix}
      e^{v1}\\[0.3em]\vdots\\[0.3em]e^{vn}\\[0.3em]
    \end{bmatrix}$

### Non vectorized implementation
```Python
u=np.zeros((n,1))
for i in range (n):
    u[i]=math.exp(v[i])
```
[[1.16788691]
 [2.55600839]
 [2.28070859]
 ...
 [1.0695791 ]
 [1.93408517]
 [1.35158682]]<br>
vectorized version:33.93435478210449 ms
### Vectorized implementation
```Python
import numpy as np

u=np.exp(v)

```
[1.16788691 2.55600839 2.28070859 ... 1.83544074 1.96388581 2.55276302]<br>
vectorized version:1.992940902709961 ms

### Other numpy functions
```Python
np.log(v)
np.abs(v)
np.maximum(v,0)
v**2
1/v
```
* np.maximum computes the element-wise maximum to take the max of every element of v with 0
*  v**2 just takes the element-wise square of each element of v.
* One over v takes the element-wise inverse

### Logistic Regression Derivatives
$J=0;
>~~dw_1=0; dw_2=0;~~  USE $dw=np.zeros((n_x,1))$

db=0;$

 for $i=1$ to $m$:

  $$z^{(i)}=w^Tx^{(i)}+b$$
  $$a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})$$
  $$J+=-[y^{(i)}log(a^{(i)})+(1-y^{(i)})log(1-a^{(i)})]$$
  $$\frac{dL}{dz^{(i)}}=dz^{(i)}=a^{(i)}-y^{(i)}$$

> 2nd loop; use **$dw+=x^{(i)}dz^{(i)}$**
>
> for $j=1$ to $n_x$:
>
>(n=2 features $[dw_1, dw_2]$. add for loop over all the features. (in this case for loop j=1 to 2))
>
>$$\frac{dL}{dw_1}+=dw_1=x_1^{(i)}z^{(i)}$$
>$$\frac{dL}{dw_2}+=dw_2=x_2^{(i)}z^{(i)}$$
>$$\vdots$$
>$$\frac{dL}{dw_j}+=dw_2=x_j^{(i)}z^{(i)}$$

$db += dz^{(i)}$

and divide them by m;

$J/=m;$

>$dw_1/=m$; $dw_2/=m$, Use $dw/=m$

db/=m$

* In order to remove the **second for loop**, we need to use vector and get rid of $dw_1=0$; $dw_2=0$;

```Python
dw=np.zeros((n_x,1))
```

### Logistic Regression Derivatives with vectorized approach
$J=0$; $dw=np.zeros((n_x,1))$; $db=0;$

for $i=1$ to $m$:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$z^{(i)}=w^Tx^{(i)}+b$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$J+=-[y^{(i)}log(a^{(i)})+(1-y^{(i)})log(1-a^{(i)})]$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\frac{dL}{dz^{(i)}}=dz^{(i)}=a^{(i)}-y^{(i)}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for k=1 to m:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$dw+=x_1^{(i)}dz^{(i)}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$dw+=x_2^{(i)}dz^{(i)}$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\vdots$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$db += dz^{(i)}$

 and divide them by m;

$J/=m;$

$dw/=m$

$db/=m$
