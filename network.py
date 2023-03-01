import numpy as np
import tensorflow as tf

class Network:
    def __init__(self, num_inputs, num_outputs, num_hidden_layers, neurons_per_hidden_layer):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.build_network()
    
    def build_network(self):
        # Define placeholders for input and output data
        self.inputs = tf.placeholder(shape=[None, self.num_inputs], dtype=tf.float32, name='inputs')
        self.targets = tf.placeholder(shape=[None, self.num_outputs], dtype=tf.float32, name='targets')

        # Define network architecture
        self.hidden_layers = []
        if self.num_hidden_layers > 0:
            self.hidden_layers.append(tf.layers.dense(self.inputs, self.neurons_per_hidden_layer, activation=tf.nn.relu))
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layers.append(tf.layers.dense(self.hidden_layers[-1], self.neurons_per_hidden_layer, activation=tf.nn.relu))
        self.outputs = tf.layers.dense(self.hidden_layers[-1] if self.num_hidden_layers > 0 else self.inputs, self.num_outputs, activation=None)

        # Define loss and optimization operators
        self.loss = tf.losses.mean_squared_error(self.targets, self.outputs)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optimizer.minimize(self.loss)

        # Define session and initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})

    def train(self, inputs, targets, batch_size=32, epochs=1):
        for epoch in range(epochs):
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                self.sess.run(self.train_op, feed_dict={self.inputs: batch_inputs, self.targets: batch_targets})

    def get_weights(self):
        weights = []
        for layer in self.hidden_layers:
            weights.append(layer.weights[0].eval(session=self.sess))
            weights.append(layer.weights[1].eval(session=self.sess))
        if self.num_hidden_layers > 0:
            weights.append(self.outputs.weights[0].eval(session=self.sess))
            weights.append(self.outputs.weights[1].eval(session=self.sess))
        return np.concatenate(weights)

    def set_weights(self, weights):
        index = 0
        for i in range(self.num_hidden_layers):
            self.hidden_layers[i].set_weights([weights[index], weights[index+1]])
            index += 2
        if self.num_hidden_layers > 0:
            self.outputs.set_weights([weights[index], weights[index+1]])

    def get_num_params(self):
        num_params = (self.num_inputs * self.neurons_per_hidden_layer) + self.neurons_per_hidden_layer
        if self.num_hidden_layers > 1:
            num_params += ((self.num_hidden_layers - 1) * (self.neurons_per_hidden_layer**2 + self.neurons_per_hidden_layer))
        if self.num_hidden_layers > 0:
            num_params += (self.neurons_per_hidden_layer * self.num_outputs) + self.num_outputs
        return num_params
