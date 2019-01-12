# Optimization algorithm -Adam

During the history of deep learning, many researchers including some very well-known researchers, sometimes proposed optimization algorithms and showed that they worked well in a few problems. But those optimization algorithms subsequently were shown not to really generalize that well to the wide range of neural networks you might want to train. So over time, the deep learning community actually developed some amount of skepticism about new optimization algorithms. And a lot of people felt that gradient descent with momentum really works well, was difficult to propose things that work much better. So, RMSprop and the Adam optimization algorithm are one of those rare algorithms that has really stood up.

Initialize $v_{dW}=0, s_{dw}=0, v_{db}=0, s_{db}=0$, then implement 'momentum' $\beta_1$ and 'RMSprop' $\beta_2$. This is commonly used algorith that is effective in many neural networks.

On iteration $t:$<br>
> Compute $dW, db$ on current mini-batch
>
> $v_{dW}=\beta_1 v_{dW} + (1-\beta_1)dW$
>
> $v_{db}=\beta_1 v_{db} + (1-\beta_1)db$$
>
> $s_{dW}=\beta_2 s_{dW} + (1-\beta_2)dW^{\overbrace{2}^{\text{element wise}}}$ $\leftarrow$ small
>
> $s_{db}=\beta_2 s_{db} + (1-\beta_2)db^{\overbrace{2}^{\text{element wise}}}$ $\leftarrow$ large

Typically, bias correction is made.

> $v_{dW}^{corrected}=\frac{v_{dW}}{(1-\beta_1^{t})}$
>
> $v_{db}^{corrected}=\frac{v_{db}}{(1-\beta_1^{t})}$
>
> $s_{dW}^{corrected}=\frac{s_{dW}}{(1-\beta_2^{t})}$
>
> $s_{db}^{corrected}=\frac{s_{db}}{(1-\beta_2^{t})}$


Then update parameters:

> $W=W-\alpha \frac{v_{dW}^{collected}}{\sqrt{s_{dW}^{corrected}+\epsilon}}$
>
> $b=b-\alpha \frac{v_{db}^{collected}}{\sqrt{s_{db}^{corrected}+\epsilon}}$

## Hyperparameters Choice:

* $\alpha$: Needs to be tuned
* $\beta_1$ 0.9 ($dW$)
* $\beta_2$ 0.999 ($dW^2$)
* $\epsilon$: Doesn't matter too much but the author of Adam paper recomends $10^{-8}$

Adam: ADAptive Moment estimation
