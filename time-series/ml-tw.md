# Machine learning on time windows

![](images/feeding-win-nn-5ee0127d.png)
if you think back to this diagram and you consider the input window to be 20 values wide, then let's call them x0, x1, x2, etc, all the way up to x19. But let's be clear. That's not the value on the horizontal axis which is commonly called the x-axis, it's the value of the time series at that point on the horizontal axis. So the value at time t0, which is 20 steps before the current value is called x0, and t1 is called x1, etc. Similarly, for the output, which we would then consider to be the value at the current time to be the y.
