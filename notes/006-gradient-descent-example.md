# Gradient Descent on m Examples
<!-- TOC -->

- [Gradient Descent on m Examples](#gradient-descent-on-m-examples)
  - [Examples](#examples)

<!-- /TOC -->
Cost function
$$J(w,b)=\frac{1}{m}\Sigma_{i=1}^mL(a^{(i)},y)$$
where
$$a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})=\sigma(w^Tx^{(i)}+b)$$

When we have one training example from $(x^{(i)},x^{(i)})$, $dw_1^{(i)}$, $dw_2^{(i)}$,$db^{(i)}$
derivative respect to say w1 of the overall cost function is also going to be the average of derivatives respect to w1 of the individual loss terms

$$\frac{d}{dw_1}J(w,b)=\frac{1}{m}\Sigma_{i=1}^m\frac{d}{dw_1}L(a^{(i),y^{(i)}})$$
where
$$\frac{d}{dw_1}L(a^{(i)},y^{(i)})=dw_1^{(i)}-(x^{(i)},y^{(i)})$$

## Examples

1. Initialize with
2.
$J=0; dw_1=0; dw_2=0; db=0;$

for $i=1$ to $m$

$$z^{(i)}=w^Tx^{(i)}+b$$
$$a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})$$
$$J+=-[y^{(i)}log(a^{(i)})+(1-y^{(i)})log(1-a^{(i)})]$$
$$\frac{dL}{dz^{(i)}}=dz^{(i)}=a^{(i)}-y^{(i)}$$
n=2 features $[dw_1, dw_2]$. add for loop over all the features. (in this case for loop j=1 to 2)
$$\frac{dL}{dw_1}+=dw_1=x_1^{(i)}z^{(i)}$$
$$\frac{dL}{dw_2}+=dw_2=x_2^{(i)}z^{(i)}$$

$$db += dz^{(i)}$$

and divide them by m;

$J/=m; dw_1/=m; dw_2/=m, db/=m$

$dz^{(i)}$ is respect to a single sample, where as $dw_1$, $dw_2$ are respect to the entire samples so $dw1=\frac{dL}{dw_1}$

* $w_1:w_1-\alpha dw_1$
* $w_2:w_2-\alpha dw_2$
* $b:b -   \alpha db$
