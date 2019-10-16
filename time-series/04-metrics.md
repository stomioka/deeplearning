# Metrics for evaluating performance

- error = forecasts - actual
- mse = np.square(errors).mean()
  -  if large errors are potentially dangerous and they cost you much more than smaller errors, then you may prefer the mse.
- rmse ( root means squared error) = np.sqrt(mse)
  -  if we want the mean of our errors' calculation to be of the same scale as the original errors, use rmse.
- mae ( mean absolute error) = np.abs(errors).mean()
  - This does not penalize large errors as much as the mse does. if your gain or your loss is just proportional to the size of the error
- mape (the mean absolute percentage error ) = np.abs(errors/x_valid).mean()

```Python
kears.metrics.mean_absolute_error(x_valid,native_forecast).numpy()
```
