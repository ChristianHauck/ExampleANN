#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

from layer import Layer
from functions import Functions
import numpy as np
import math


class MLP:

    def __init__(self, dimensions): # first element = input-dim, the following = layer-dims
        self.num_layers = len(dimensions) - 1   # -1 : subtract input-dim
        self.layers = []
        # and now the specified layers
        for i in range(self.num_layers):
            num_in = dimensions[i]
            num_out = dimensions[i+1]
            self.layers.append(Layer(num_in, num_out))
        # activation functions
        self.activation_str = None
        self.functions = Functions()
        return


    def init_layers(self):
        for layer in self.layers:
            rnd_wght = (np.random.rand(layer.num_out, layer.num_in) * 2.0 - 1.0) / 100
            layer.init_weights(rnd_wght)


    def set_activation(self, activation_str):
        self.activation_str = activation_str


    def activation(self, vector):
        result = self.functions.get_fu[self.activation_str](vector)
        return result


    def deriv_act(self, vector):
        result = self.functions.get_deriv[self.activation_str](vector)
        return result


    def learn(self, m_x, m_y, learning_rate, max_loops=1000):
        loop_nr = 0
        while True:
            error = 0.0
            for ex in range(len(m_x)):                  # for each example - index: ex
                # Just copy the input into INPUT of layer 0
                self.layers[0].input = m_x[ex]

                # Now propagate a[0] forward and store the interim results a[l] for the layers l
                for l in range(self.num_layers):
                    self.layers[l].sum = np.dot(self.layers[l].weights, self.layers[l].input)
                    self.layers[l].output = self.activation(self.layers[l].sum)
                    if (l < self.num_layers - 1):
                        self.layers[l + 1].input = self.layers[l].output
                        # self.layers[l + 1].input = np.copy(self.layers[l].output)

                # Compute the error at the output layer
                out_layer = self.layers[self.num_layers - 1]  # output layer
                # m_y is not an mdarray! ToDo!
                out_layer.delta = self.deriv_act(out_layer.sum) * (m_y[ex] - out_layer.output) # ex = example index

                # Compute the other deltas, backwards running through the layers
                for l in reversed(range(self.num_layers - 1)):
                    layer_m = self.layers[l]
                    layer_m_plus_1 = self.layers[l+1]
                    delta_m = self.deriv_act(layer_m.sum) * \
                              np.dot(layer_m_plus_1.weights.T, layer_m_plus_1.delta)
                    layer_m.delta = delta_m

                # And finally adjust weights.
                # This can be done in forward order because all data are available here.
                for l in range(self.num_layers):
                    layer = self.layers[l]
                    for i in range(layer.num_out):
                        delta_w = learning_rate * layer.delta[i] * layer.input
                        row = layer.weights[i] + delta_w
                        layer.weights[i] = row
                        # layer.weights[j] = layer.weights[j] + delta_w

                # Calculate a more or less reasonable error metric
                error += Functions.squared_error(m_y[ex], self.layers[self.num_layers - 1].output)

            # Calculate a more or less reasonable error metric
            error /= len(m_x)
            loop_nr += 1

            if error < 1.0E-9 or loop_nr >= max_loops * len(m_x):
                print("learning loops: ", loop_nr / len(m_x))
                break  # while

        return


    def run(self, x):
        self.layers[0].input = x
        for l in range(self.num_layers):
            self.layers[l].sum = np.dot(self.layers[l].weights, self.layers[l].input)
            # self.layers[l].output = self.activation(self.layers[l].sum)
            self.layers[l].output = self.activation(self.layers[l].sum)
            if (l < self.num_layers - 1):
                self.layers[l+1].input = self.layers[l].output
        #
        return self.layers[self.num_layers - 1].output


    def show(self, out_str, m_x, m_y):
        print(out_str)
        # for i in range(self.num_layers):
        #     print("weigt matrix: ", self.layers[i].weights)
        for i in range(len(m_x)):
            result = self.run(m_x[i])
            # print("input ", m_x[i], "desired output ", m_y[i], " output ", result)
            #print("input ", m_x[i])
            print("desired output ", m_y[i])
            print("output ", result)


if __name__ == '__main__':
    mlp = MLP([3,5,2])
    mlp.init_layers()
    mlp.init_activation("sigmoid")
