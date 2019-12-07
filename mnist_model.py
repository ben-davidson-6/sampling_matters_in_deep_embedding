import tensorflow as tf
import tensorflow.estimator as estimator
from margin_loss import margin_loss, PAIRWISE_DISTANCES


class MNISTModel():

    def embedding_net(self, features):
        features = tf.reshape(features, [-1, 28*28])
        net = tf.layers.dense(features, 512, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=None, use_bias=False)
        net = tf.nn.l2_normalize(net, axis=1)
        return net

    def model_fn(self, features, labels, mode, params):
        tf.summary.image('image', features)
        embedding = self.embedding_net(features)

        if mode == estimator.ModeKeys.PREDICT:
            return self.predict_spec(embedding)

        betas = tf.get_variable('beta_margins', initializer=params['beta_0'] * tf.ones([10]))
        loss = margin_loss(labels, embedding, betas, params)
        if mode == estimator.ModeKeys.TRAIN:
            return self.train_spec(loss, params)
        if mode == estimator.ModeKeys.EVAL:
            return self.eval_spec(loss, labels)

    #####################################################################
    # Defining train/eval/predict for estimator
    #####################################################################

    def predict_spec(self, features):
        named_predictions = {
            'embedding': features}
        return estimator.EstimatorSpec(
            estimator.ModeKeys.PREDICT,
            predictions=named_predictions)

    def train_spec(self, loss, params):
        train_op = self._training_op(loss, params)
        return estimator.EstimatorSpec(
            estimator.ModeKeys.TRAIN,
            loss=loss,
            train_op=train_op)

    def eval_spec(self, loss, labels):
        g = tf.get_default_graph()
        D = g.get_tensor_by_name(PAIRWISE_DISTANCES + ':0')
        D *= -1
        _, top_1 = tf.nn.top_k(D, 2)
        top_1 = top_1[:, 1]
        estimated = tf.gather_nd(labels, top_1[:, None])

        # Define the metrics:
        metrics_dict = {
            'Map@1': tf.metrics.accuracy(labels, estimated)}

        # return eval spec
        return estimator.EstimatorSpec(
            estimator.ModeKeys.EVAL,
            loss=loss,
            eval_metric_ops=metrics_dict)

    #####################################################################
    # Traingin op
    #####################################################################

    def _training_op(self, loss, params):
        learning_rate = params['initial_lr']
        tf.summary.scalar('learning_rate', learning_rate)
        optimiser = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=params['momentum'],)
        return optimiser.minimize(loss, global_step=tf.train.get_or_create_global_step())


class Mnist():

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def neccessary_processing(self, image, label):
        image = tf.reshape(image, [28, 28, 1])
        image = image/255
        image -= tf.constant([0.5])[None, None]
        image /= tf.constant([0.25])[None, None]
        return image, label

    def train_generator(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        def gen():
            for image, label in zip(x_train, y_train):
                yield image, label
        return gen

    def build_training_data(self):
        gen = self.train_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28, 28), ()))
        ds = ds.shuffle(5000).repeat()
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds

    def validation_generator(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        def gen():
            for image, label in zip(x_test, y_test):
                yield image, label
        return gen

    def build_validation_data(self):
        gen = self.validation_generator()
        ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32), ((28, 28), ()))
        ds = ds.shuffle(1000)
        ds = ds.map(self.neccessary_processing, num_parallel_calls=4)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(1)
        return ds
