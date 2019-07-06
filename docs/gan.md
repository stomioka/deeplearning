## GAN
The generator G aim is to capture the data distribution, while the discriminator D estimates the probability that a sample came from the training data rather than G.

![](images/gananormalydetect-f3345b7c.png)

### Notations:
- $p_g$: generator’s noize distribution over data $x$
- $p_z(z)$: input prior noise variable. $z$ is each noise samples
- $G$: Differentiable function represented by a multilayer perceptron with parameters $\theta_g$. It takes $z$ and distribute to $p_g$.
- $G(z; θ_g)$: mapped data space where $θ_g$ are the generator parameters
- $D(x; θ_d)$: Discriminator where $θ_D$ are discriminator parameters. This is the second multilayer perceptron which outputs a single scalar (1 or 0).
- $D(x)$: the probability that $p_x$ (1) came from the data rather than $p_g$ (0)


To learn a generative distribution $p_g$ over the data $x$ the generator builds a mapping from a prior noise distribution $p_z(z)$ to a data space as $G(z; θ_g)$ to minimize $log(1 − D(G(z)))$.
.
The discriminator outputs a single scalar representing the probability that x came from real data rather than from $p_g$.


$$\min_G \max_D V(D, G)=
\mathbb{E}_{x\sim p_{data}(x)}[\log D(x)]
+ \mathbb{E}_{z\sim p_z(z)}[\log(1 - D(G(z)))]$$

In this context, we want to maximize $V(D,G)$ with respect to the discriminator $D$, that is, the $max_DV(D,G)$ part from equation 1 and minimize $log(1 − D(G(z)))$.
So if $D$ can classify sample $x$ correctly, $logD(x)$ gets larger, and if $G$ can generates sample as close as sample $x$, then $D(G(x))$ becomes closer to 1 so $log(1 − D(G(z)))$ gets closer to 0.

Algorithm from the original paper

![](images/gananormalydetect-88de51d6.png)



## Reference:
- Goodfellow, I.  et al.,  [Generative Adversarial Nets.](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) pp. 2672–2680, 2014.
