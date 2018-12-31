# Building blocks of deep neural networks


![](images/023-building-block-deep-neural-network-c21152d1.png)


Layer $l$: $w^{[l]}$, $b^{[l]}$

$\color{blue}{\text{Forward: Input }} a^{[l-1]}\color{blue}{\text{, Output }}a^{[l]}$

$\color{green}{z^{[l]} : w^{[l]}a^{[l-1]}+b}$ $\color{blue}{\text{cache: } z^{[l]}}$

$\color{green}{a^{[l]} : g(z^{[l]})}$

$\begin{matrix}\color{blue}{\text{Backward: Input }} &da^{[l]}&\color{blue}{\text{ Output }}&da^{[l-1]}\\
&cache(z^{[l]})&&dw^{[l]}\\
&&&db^{[l]}\end{matrix}$

![](images/023-building-block-deep-neural-network-c5ba613f.png)

Forward and backward functions
![](images/023-building-block-deep-neural-network-b2ebd4b9.png)
