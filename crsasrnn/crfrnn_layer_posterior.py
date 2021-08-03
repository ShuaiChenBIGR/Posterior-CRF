"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import Modules.Networks.CRF.CRFasRNN_tensorflow.CRFasRNNLayer.lattice_filter_op_loader2_2_posterior as lattice_filter_op_loader2

custom_module2 = lattice_filter_op_loader2.module


class New_CrfRnnLayer_3d_GPU_2_2_posterior(Layer):
    """Implemented by Shuai

    Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, trainable, **kwargs):
        self.image_dims = image_dims
        self.num_classes = num_classes
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        self.trainable = trainable
        self.spatial_ker_weights = None
        self.bilateral_ker_weights = None
        self.compatibility_matrix = None
        super(New_CrfRnnLayer_3d_GPU_2_2_posterior, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        #tf.initializers.constant(value=[[300, 0, 0], [0, 300, 0], [0, 0, 300]])
        #tf.initializers.constant(value=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        #'uniform'
        self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
                                                   shape=(self.num_classes,),
                                                   initializer=tf.initializers.constant(value=[[1.41, 1.41]]),
                                                   trainable=True)

        self.spatial_ker_weights = tf.diag(self.spatial_ker_weights)

        # self.spatial_ker_weights = self.add_weight(name='spatial_ker_weights',
        #                                            shape=(self.num_classes, self.num_classes),
        #                                            initializer=tf.initializers.constant(value=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        #                                            trainable=False)
        # Weights of the bilateral kernel
        # self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
        #                                              shape=(self.num_classes, self.num_classes),
        #                                              initializer=tf.initializers.constant(value=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        #                                              trainable=False)
        #
        self.bilateral_ker_weights = self.add_weight(name='bilateral_ker_weights',
                                                     shape=(self.num_classes,),
                                                     initializer=tf.initializers.constant(value=[[3.85, 3.85]]),
                                                     trainable=True)
        self.bilateral_ker_weights = tf.diag(self.bilateral_ker_weights)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer=tf.initializers.constant(value=[[1.0, 0.0], [0.0, 1.0]]),
                                                    trainable=True)

        super(New_CrfRnnLayer_3d_GPU_2_2_posterior, self).build(input_shape)

    def call(self, inputs):
        unaries = tf.expand_dims(inputs[0][0, :, :, :, :], axis=0)

        rgb = tf.expand_dims(inputs[1][0, :, :, :, :], axis=0)

        unaries_shape = unaries.get_shape()
        all_ones = tf.Variable(np.ones(unaries_shape, dtype=np.float32), dtype=tf.float32)
        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module2.lattice_filter_posterior(all_ones, rgb, bilateral=False, theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module2.lattice_filter_posterior(all_ones, rgb,
                                                           bilateral=True, theta_alpha=self.theta_alpha, theta_beta=self.theta_beta)
        q_values = unaries

        for i in range(self.num_iterations):
            # q_values = tf.nn.softmax(q_values)
            # softmax_out = q_values

            # Spatial filtering
            spatial_out = custom_module2.lattice_filter_posterior(q_values, rgb, bilateral=False, theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals

            # Bilateral filtering
            bilateral_out = custom_module2.lattice_filter_posterior(q_values, rgb, bilateral=True, theta_alpha=self.theta_alpha,
                                                         theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals

            # Weighting filter outputs
            message_passing = tf.matmul(self.spatial_ker_weights,
                                         tf.transpose(tf.reshape(spatial_out, (-1, self.num_classes)))) + tf.matmul(self.bilateral_ker_weights,
                                         tf.transpose(tf.reshape(bilateral_out, (-1, self.num_classes))))

            # Compatibility transform
            pairwise = tf.matmul(self.compatibility_matrix, message_passing)

            # Adding unary potentials
            pairwise_trans = tf.transpose(pairwise)
            pairwise = tf.reshape(pairwise_trans, unaries_shape)

            nsmall = tf.constant(1.0, shape=unaries.get_shape())
            pairwise = tf.multiply(nsmall, pairwise)

            # q_values = unaries - pairwise + pairwise
            q_values = unaries - pairwise
            # q_values = - pairwise
        return q_values

    def compute_output_shape(self, input_shape):
        return input_shape

