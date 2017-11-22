#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell


class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. To locate the TensorFlow installation path, do
    the following:

    1. In Python, type 'import tensorflow as tf', then 'print(tf.__file__)'

    2. According to the output, find tensorflow_install_path/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.

    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step
                         to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################

        self.num_units=int(num_units)
        self.num_proj=int(num_proj)
        self.forget_bias=forget_bias
        

        if activation is None:
            self.activation = tf.tanh
        else:
            self.activation=activation

        xav_init = tf.contrib.layers.xavier_initializer

        self.W = tf.get_variable('W', shape=[4, 1, self.num_units], initializer=xav_init())

        self.WFinal=tf.get_variable('WFinal', shape=[self.num_units, num_proj], initializer=xav_init())

        self.U = tf.get_variable('U', shape=[4, 2, self.num_units], initializer=xav_init())


    


        # raise NotImplementedError('Please edit this function.')

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################

        return self.num_units + self.num_proj

        # raise NotImplementedError('Please edit this function.')

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################

        return self.num_proj

        # raise NotImplementedError('Please edit this function.')

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the
        very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################


        num_units=self.num_units
        num_proj=self.num_proj

        # print(state.get_shape())


        c=tf.slice(state, [0, 0], [-1, num_units])
        h=tf.slice(state, [0, num_units], [-1, num_proj])

        # print(c.get_shape())


        U=self.U
        W=self.W
        WFinal=self.WFinal
        # print(self.num_units)
        # print(state.get_shape())

        # print(type(inputs))
        # print("inputs", inputs.get_shape())
        # print("W", W[0].get_shape())
        # print("h", h.get_shape())
        # print("U", U[0].get_shape())
        
        # c,h = tf.unstack(state)
        ####
        # GATES
        #
        #  input gate
        i = tf.sigmoid(tf.matmul(inputs,W[0]) + tf.matmul(h,U[0]))
        #  forget gate
        f = tf.sigmoid(tf.matmul(inputs,W[1]) + tf.matmul(h,U[1]) + self.forget_bias)
        #  output gate
        o = tf.sigmoid(tf.matmul(inputs,W[2]) + tf.matmul(h,U[2]))
        # gate weights - candidate valuesU    
        g = tf.tanh(tf.matmul(inputs,W[3]) + tf.matmul(h,U[3]))
        ###
        # new internal cell state

        c = c*f + g*i

        # print(c.get_shape())
        # output state
        h = tf.matmul(self.activation(c)*o, WFinal)

        # print(h.get_shape())

        state_final=tf.concat([c, h], 1)

        # return tf.stack([h, c])
        return (h,state_final)

        # raise NotImplementedError('Please edit this function.')
