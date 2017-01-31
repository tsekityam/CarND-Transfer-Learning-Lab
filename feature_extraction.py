import pickle
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('batch_size', 128, "Number of samples per gradient update")
flags.DEFINE_integer('nb_epoch', 10, "The number of epochs to train the model")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    nb_classes = len(np.unique(y_train))

    a = Input(shape=X_train[0].shape)
    b = Flatten()(a)
    b = Dense(nb_classes, activation='softmax')(b)
    model = Model(input=a, output=b)

    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

    # train your model here
    history = model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.nb_epoch, validation_data=(X_val, y_val))

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
