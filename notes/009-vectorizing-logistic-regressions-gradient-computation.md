# Vectorizing Logistic Regression's Gradient Output
<!-- TOC -->

- [Vectorizing Logistic Regression's Gradient Output](#vectorizing-logistic-regressions-gradient-output)
  - [Vectorization to avoid For Loop](#vectorization-to-avoid-for-loop)
    - [For Loop approach](#for-loop-approach)
    - [Vectorization Approach](#vectorization-approach)

<!-- /TOC -->
In the gradient descent computation, you would do the followings:

$dz^{(1)}=a^{(1)}-y^{(1)}$

$dz^{(2)}=a^{(2)}-y^{(2)}$

$dz^{(3)}=a^{(3)}-y^{(3)}$

$\vdots$

so, $dZ=\begin{bmatrix}dz^{(1)}&dz^{(2)}& \cdots& dz^{(m)} \end{bmatrix}$ and this is $\mathbb{R}^{1,m}$


Recall $A=\begin{bmatrix}a^{(1)}&a^{(2)}& \cdots& a^{(m)} \end{bmatrix}$
and $Y=\begin{bmatrix}y^{(1)}&y^{(2)}& \cdots& y^{(m)} \end{bmatrix}$, so

⭐  $dZ=A-Y=\begin{bmatrix}a^{(1)}-y^{(1)}&a^{(2)}-y^{(2)}& \cdots& a^{(m)}-y^{(m)} \end{bmatrix}$

We had the second for loop for:

---
$dw=0$

$dw+=x^{(1)}dz^{(1)}$

$dw+=x^{(2)}dz^{(2)}$

$\vdots$

$dw+=x^{(m)}dz^{(m)}$


$dw/=m$

This for loop can be written as

⭐ $dw=XdZ^T$

$=\frac{1}{m}\begin{bmatrix}
|       &     |   &        &    |   \\
x^{(1)} & x^{(2)} & \ldots & x^{(m)}\\
|       &     |   &        &    |   \\
\end{bmatrix}\begin{bmatrix} dz^{(1)}\\
dz^{(2)} \\
\vdots \\
dz^{(m)} \\
\end{bmatrix}$

$=\frac{1}{m}
\begin{bmatrix}
x^{(1)}dz^{(1)} & x^{(2)}dz^{(2)} & \cdots & x^{(m)}dz^{(m)}
\end{bmatrix}$

---
$db=0$

$db+=dz^{(1)}$

$db+=dz^{(2)}$

$\vdots$

$db+=dz^{(m)}$

$db/=m$

This for loop can be written as

⭐ $db=\frac{1}{m}\Sigma_{i=1}^mdz^{(i)}$

so in python you written
```Python
db=np.sum(dZ)/m
```


## Vectorization to avoid For Loop

### For Loop approach
$J=0$; $dw=np.zeros((n_x,1))$; $db=0;$

for $i=1$ to $m$:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$z^{(i)}=w^Tx^{(i)}+b$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$J+=-[y^{(i)}log(a^{(i)})+(1-y^{(i)})log(1-a^{(i)})]$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\frac{dL}{dz^{(i)}}=dz^{(i)}=a^{(i)}-y^{(i)}$

>for k=1 to m:
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$dw+=x_1^{(i)}dz^{(i)}$
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$dw+=x_2^{(i)}dz^{(i)}$
>
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\vdots$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$db += dz^{(i)}$

 and divide them by m;

`$J/=m;$`

`$dw/=m$`

`$db/=m$`

### Vectorization Approach
`$J=0$; $dw=np.zeros((n_x,1))$; $db=0;$`

$Z=wTX+b$

```Python
Z=np.dot(w.T,x)+b
```

$A=\sigma({Z})$

$dZ=A-Y$

$dw=\frac{1}{m}XdZ^T$

$db=\frac{1}{m}\Sigma_{i=1}^m(dz)$

```Python
db=np.sum(dz)/m
```

Then parameters w and b are updated as
* $w_1:w_1-\alpha dw_1$
* $w_2:w_2-\alpha dw_2$
* $b:b -   \alpha db$
