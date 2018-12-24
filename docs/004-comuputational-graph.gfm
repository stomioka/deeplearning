# Computation Graph

<!-- TOC -->

- [Computation Graph](#computation-graph)
- [Derivatives with  Computation graph](#derivatives-with--computation-graph)
  - [Derivative of $J$ with respect to $v$.](#derivative-of-j-with-respect-to-v)
  - [Derivative of $J$ with respect to $a$.](#derivative-of-j-with-respect-to-a)
  - [Derivative of $J$ with respect to $u$](#derivative-of-j-with-respect-to-u)
  - [Derivative of $J$ with respect to $b$](#derivative-of-j-with-respect-to-b)
  - [Derivative of $J$ with respect to $c$](#derivative-of-j-with-respect-to-c)

<!-- /TOC -->
Let's use a simple function, $J(a, b, c)=3(a+bc)$ to illustrate a computation graph.

Computing this function has 3 distinct steps.
1. Compute $u=bc$
2. Compute $v=a+u$
3. Compute $J=3v$

![](images/computationgraph.svg)

From top to bottom, the value of J can be computed. In order to calculate derivatives, you will calculate from bottom to top.

*generated with digraph
```
G:\GoogleDrive\deeplearning>diagrams dot computationgraph.dot images/computationgraph.svg
```

# Derivatives with  Computation graph

![](images/computationgraph.svg)

## Derivative of $J$ with respect to $v$.

$\frac{dJ}{dv}$

If we were to nudge the value $v$ a litte bit, how would the value of J change?
$J=3v$ and $v=11$, so if $v=11.001$ then $J=33.003$, so $J$ goes 3 times up, so $\frac{dJ}{dv}=3$.
To compute derivative of $J$ with respect to v, we went one step backwards to $v$.

## Derivative of $J$ with respect to $a$.

$\frac{dJ}{da}$

If we were to nudge the value $a$ a litte bit, how would the value of $J$ change?
$J=3v$ and $a=5$, so if $a=5.001$ then $v=5.001+u=5.001+6=11.001$, and $J=33.003$, so $J$ goes 3 times up, so $\frac{dJ}{da}=3$.
To compute derivative of $J$ with respect to a, we went two step backwards to v.

In calculus,  $\frac{dJ}{da}=3$ is the product of how much changes to J and v, and it can be written as
 $\frac{dJ}{da}=3=\frac{dJ}{dv}*\frac{dv}{da}$. this is called chain rule. IN this example, $\frac{dv}{da}=1$ and $\frac{dJ}{dv}=3$ so $1*3=3$

 Final output in this case is $J$

 $\frac{dFinalOutputVar}{dvar}$ where $dFinalOutputVar = J$ and $dvar \in (a, b, c, u, v)$

 ### Notation:
* $\frac{dFinalOutputVar}{dvar}$ -> "dvar"
*  "dvar" = The derivative of a final output variable with respect to various intermediate quantities.
* $\frac{dJ}{dv}$ -> "dv"
* $\frac{dJ}{da}$ -> "da"

## Derivative of $J$ with respect to $u$
![](images/computationgraph.svg)
$\frac{dJ}{du}$ (or $du$)

* $u=6$ -> 6.001
* $v=11$ -> 11.001
* $J=3v$ -> 33.00**3**
* so $\frac{dJ}{du}=du=3=\frac{dJ}{dv}*\frac{dJ}{du}=3*1=3$

## Derivative of $J$ with respect to $b$
With chain rule,
* so $\frac{dJ}{db}=db=\frac{dJ}{du}*\frac{du}{db}$
* $b=3$ -> 3.001
* $u=6$ -> 6.002

* so \frac{du}{db}=2
* $\frac{dJ}{db}==\frac{dJ}{du}*\frac{du}{db}=3*2=6$

## Derivative of $J$ with respect to $c$
* $\frac{dJ}{dc}==\frac{dJ}{du}*\frac{du}{dc}=3*3=9$

To compute derivative in the computational graph, calculate from the bottom to the top.
