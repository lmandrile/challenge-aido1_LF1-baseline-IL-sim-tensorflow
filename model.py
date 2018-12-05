import numpy as np
import tensorflow as tf
import cv2

from _layers import one_residual


class TensorflowModel:
    def __init__(self, observation_shape, action_shape, graph_location, seed=1234):
        # model definition
        self._observation = None
        self._action = None
        self._computation_graph = None
        self._optimization_op = None

        self.tf_session = tf.InteractiveSession()

        # restoring
        self.tf_checkpoint = None
        self.tf_saver = None

        self.seed = seed

        self._initialize(observation_shape, action_shape, graph_location)

    def steering_to_wheel(self, action):
        gain=1.0
        trim=0.0
        radius=0.0318
        k=27.0
        limit=1.0
        wheel_dist=0.102
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = k
        k_l = k

        # adjusting k by gain and trim
        k_r_inv = (gain + trim) / k_r
        k_l_inv = (gain - trim) / k_l

        omega_r = (vel + 0.5 * angle * wheel_dist) / radius
        omega_l = (vel - 0.5 * angle * wheel_dist) / radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, limit), -limit)
        u_l_limited = max(min(u_l, limit), -limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

    def predict(self, state):
        #state = state[136:480, 0:640]
        #state = cv2.resize(state, (640, 480))
        
        angle = self.tf_session.run(self._computation_graph, feed_dict={
            self._observation: [state],
        })
        action = (0.5, angle)
        action = self.steering_to_wheel(action)
        return np.squeeze(action)

    def train(self, observations, actions):
        _, loss = self.tf_session.run([self._optimization_op, self._loss], feed_dict={
            self._observation: observations,
            self._action: actions
        })
        return loss

    def commit(self):
        self.tf_saver.save(self.tf_session, self.tf_checkpoint)

    def computation_graph(self):
        model = one_residual(self._preprocessed_state, seed=self.seed, filter_amount = 128)
        model = tf.layers.dense(model, units=200, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))
        model = tf.layers.dense(model, units=300, activation=tf.nn.relu,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed),
                                bias_initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed))

        #model = tf.layers.dense(model, self._action.shape[1])
        model = tf.layers.dense(model, 1)

        return model

    def _optimizer(self):
        return tf.train.AdamOptimizer()

    def _loss_function(self):
        return tf.losses.mean_squared_error(self._action, self._computation_graph)

    def _initialize(self, input_shape, action_shape, storage_location):
        if not self._computation_graph:
            self._create(input_shape, action_shape)
            self._storing(storage_location)
            self.tf_session.run(tf.global_variables_initializer())

    def _pre_process(self):
        resize = tf.map_fn(lambda frame: tf.image.resize_images(frame, (60, 80)), self._observation)
        and_standardize = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), resize)
        self._preprocessed_state = and_standardize

    def _create(self, input_shape, output_shape):
        self._observation = tf.placeholder(dtype=tf.float32, shape=input_shape, name='state')
        self._action = tf.placeholder(dtype=tf.float32, shape=output_shape, name='action')
        self._pre_process()

        self._computation_graph = self.computation_graph()
        self._loss = self._loss_function()
        self._optimization_op = self._optimizer().minimize(self._loss)

    def _storing(self, location):
        self.tf_saver = tf.train.Saver()

        self.tf_checkpoint = tf.train.latest_checkpoint(location)
        if self.tf_checkpoint:
            self.tf_saver.restore(self.tf_session, self.tf_checkpoint)
        else:
            self.tf_checkpoint = location

    def close(self):
        self.tf_session.close()
