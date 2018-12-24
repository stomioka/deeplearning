# A note on python/numpy vectors
<!-- TOC -->

- [A note on python/numpy vectors](#a-note-on-pythonnumpy-vectors)
  - [rank 1 array](#rank-1-array)

<!-- /TOC -->
## rank 1 array
```Python
import numpy as np

a=np.random.randn(5)
print(a)
```
```
[-1.0075852  -0.28675951  0.42515959  0.26298535  1.35111449]
```

Above is rank 1 array
```Python
print(a.shape)
```
```
(5,)
```
```Python
print(a.T)
```
It gives a single number
```
3.1728912782704035
```

If you explicly specify the shape of the vector

```Python
a=np.random.randn(5,1)
print(a)

```
You gradient
```
[[-0.40628911]
 [ 1.07310438]
 [-0.86370404]
 [-0.65018251]
 [ 0.26537687]]
 ```
 If you transpose it, you get

 ```
 print(a.T)
 ```

[[-0.40628911  1.07310438 -0.86370404 -0.65018251  0.26537687]]

and whe you use a dot function, you get

```python
print(np.dot(a,a.T))
```
```
[[ 0.16507084 -0.43599062  0.35091354  0.26416207 -0.10781973]
 [-0.43599062  1.15155301 -0.92684459 -0.6977137   0.28477708]
 [ 0.35091354 -0.92684459  0.74598466  0.56156526 -0.22920707]
 [ 0.26416207 -0.6977137   0.56156526  0.42273729 -0.1725434 ]
 [-0.10781973  0.28477708 -0.22920707 -0.1725434   0.07042488]]
```

You can use assert

```Python
assert(a.shape ==(5,1))
```
