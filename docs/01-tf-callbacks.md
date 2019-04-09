# Using Callbacks to control training

```Python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={})
    if(longs.get('loss')<0.05):
      print('n\Loss is low so cancelling training!')
      self.model.stop_training=True
```

```python
import tensorflow as tf
callbacks=myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images  = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
                                    model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
```

Callback on mnist FFNN
```Python
import tensorflow as tf
mnist = tf.keras.datasets.mnist
class mnistCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('acc')>0.99):
      print('stopped because acc is >0.99')
      self.model.stop_training=True




(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test=x_train/255.0, x_test/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

callbacks=mnistCallback()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10, callbacks=[callbacks])
```
