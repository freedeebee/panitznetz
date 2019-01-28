import tensorflow as tf
import util
from tensorflow.python.keras.initializers import random_normal

print(tf.__version__)



def get_model(hidden_units,
              learning_rate,
              std_dev,
              activation):
    ''' creates a neural network '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_units,
                              activation=activation,
                              kernel_initializer=random_normal(stddev=std_dev)),
        tf.keras.layers.Dense(2,
                              activation=tf.nn.sigmoid,
                              kernel_initializer=random_normal(stddev=std_dev)),
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate), 
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


def train(dataset, model):
    ''' trains the neural network on a dataset '''

    X,t = dataset.get(batchsize=None)

    for epoch in range(1000):
        model.fit(X, t, epochs=1)
        util.plot(model, X, t, 'training.pdf')


def main_circle():
    dataset = util.DatasetCircle()
    model = get_model(hidden_units=10,
                      learning_rate=1.0, #1.0,
                      std_dev=1.0,
                      activation=tf.nn.sigmoid)
    train(dataset, model)


def main_logo():
    dataset = util.DatasetLogo(n=3000)
    model = get_model(hidden_units=200,
                      learning_rate=0.1,
                      std_dev=1.0,
                      activation=tf.nn.tanh)
    train(dataset, model)


if __name__ == "__main__":
    main_circle()
