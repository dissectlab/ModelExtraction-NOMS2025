import tensorflow as tf
import numpy as np

(x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x = x.astype('float32') / 255.0

def T(y): return tf.image.resize(y, [16, 16])

x = T(x)

with s.scope():
    class Q(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.u = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(16, 16, 3)),
                tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2),
                tf.keras.layers.Conv2D(8, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2)
            ])
            self.v = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(8, 3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')
            ])
        def call(self, z): return self.v(self.u(z))

    m = Q()
    m.compile(optimizer='adam', loss='mse')
    m.fit(x, x, epochs=3, batch_size=128, verbose=0)

    idx = np.random.choice(len(x), 1)
    out = m(x[idx])
    print(out.shape)
