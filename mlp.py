#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>

import numpy as np
from layer import Layer
from functions import Functions
import math


class MLP():

    def __init__(self, dimensions): # first element = input-dim, the following = layer-dims
        self.num_layers = len(dimensions) - 1   # -1 : subtract input-dim
        self.layers = []
        # and now the specified layers
        for i in range(self.num_layers):
            num_in = dimensions[i]
            num_out = dimensions[i+1]
            self.layers.append(Layer(num_in, num_out))

        return


    def init_layers(self, fu_name):
        for l in self.layers:
            m_weights = np.random.rand(l.num_nodes, l.num_in) / 1
            # m_weights = np.full((l.num_nodes, l.num_in), 0.0)
            l.init_weights(m_weights)

            # overwrite default-activation set at layer-construction in __init__
            l.init_activation(fu_name)


    def test_encoder(self, learn_rate, num_loops):
        # Inputs
        M_in = np.array([
            [-1,0,0],
            [-1,0,1],
            [-1,1,0],
            [-1,1,1]
            ])
        v_target = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.6, 0.6],
            [0.9, 0.8],
            ])

        #Initialize
        self.init_layers('sigmoid')
        self.show("before training", M_in, v_target)  # before training

        # training
        self.learn(
            M_in,       # input matrix/vectors
            v_target,   # output target vector of "or" function
            learn_rate, # learning rate
            num_loops   # number of loops
            )

        self.show("after training", M_in, v_target)  # after training

        # Now test an input vector unknown to the net
        arr = np.array([-1, 0.8, 0.7])
        print('Testing network with input ', arr, ' : ', self.run(arr))

        return


    def learn(self, M_x, v_y, learning_rate, max_loops=1000):
        loop_nr = 0
        while True:
            for i in range(M_x.shape[0]):                  # for each example - index: i
                # Just copy the input into INPUT of layer 0
                self.layers[0].IN = M_x[i]

                # Now propagate a[0] forward and store the interim results a[l] for the layers l
                for l in range(self.num_layers):    # for each layer except of the first - index l
                    layer = self.layers[l]
                    y = np.empty((layer.num_nodes))
                    res = np.empty((layer.num_nodes))
                    for j in range(layer.num_nodes):    # for each row in weight matrix of layer l - index j
                        res[j] = np.dot(layer.weight_matrix[j], layer.IN)
                        y[j] = layer.activation(res[j])

                    layer.OUT = y       # = g(RAW_OUT) = g(SUM(wv))
                    layer.RAW_OUT = res # = SUM(wv)
                    if l < self.num_layers - 1:
                        self.layers[l + 1].IN = self.layers[l].OUT

                # Compute the error at the output layer
                out_layer = self.layers[self.num_layers - 1]  # output layer
                out_layer.DELTA = out_layer.deriv_act(out_layer.RAW_OUT) * (v_y[i] - out_layer.OUT) # i = example index

                # Compute the other deltas, backwards running through the layers
                for l in reversed(range(self.num_layers - 1)):
                    layer_m = self.layers[l]
                    layer_m_plus_1 = self.layers[l+1]
                    delta_m = layer_m.deriv_act(layer_m.RAW_OUT) * \
                                   np.dot(layer_m_plus_1.weight_matrix.T, layer_m_plus_1.DELTA)
                    layer_m.DELTA = delta_m

                # And finally adjust weights.
                # This may be done in forward order because all data are available here.
                for l in range(self.num_layers):
                    layer = self.layers[l]
                    for j in range(layer.num_nodes):
                        delta_w = learning_rate * layer.DELTA[j] * layer.IN
                        layer.weight_matrix[j] = layer.weight_matrix[j] + delta_w

            # Calculate a more or less reasonable error metric
            error = Functions.squared_error(v_y[i], self.layers[self.num_layers - 1].OUT)
            error = math.sqrt(error)
            loop_nr += 1

            if error < 0.001 or loop_nr > max_loops:
                print("learning loops: ", loop_nr)
                break  # while

        return


    def run(self, x):
        for i in range(self.num_layers):
            y = self.layers[i].run_wo_thr(x)
            x = y
        return y


    def show(self, out_str, M_in, v_desired_out):
        print(out_str)
        for i in range(self.num_layers):
            print("weigt matrix: ", self.layers[i].weight_matrix)
        for i in range(M_in.shape[0]):
            result = self.run(M_in[i])
            print("input ", M_in[i], "desired output ", v_desired_out[i], " output ", result)



if __name__ == '__main__':
    # Main entry point for tests
    mlp = MLP([3,5,2])
    mlp.test_encoder(10, 1000)