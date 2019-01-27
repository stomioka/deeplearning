# Simple Convolutional Network

#Example
![](images/083-simple-cnn-2eb8aae5.png)

We have an image x and decide is this a cat of not (0 or 1).

Input Image:$x$

$n_H^{[0]}=n_W^{[0]}=39$<br>
$n_c^{[0]}=3$

10 filters<br>
$f^{[1]}=3$<br>
$s^{[1]}=1$<br>
$p^{[1]}=0$

Output Image: $a^{[1]}$<br>
$\frac{n+2p-f}{s}+1=37$ so the output is $37\times 37 \times 10$

$n_H^{[1]}=n_W^{[1]}=37$<br>
$n_c^{[1]}=10$

Repeat this process. Finally $a^{[3]}$ gets flattened to 1960 units.

## Types of layer in a convolutional network

1. Convolution (Conv)
2. Pooling (POOL)
3. Fully Connected (FC)
