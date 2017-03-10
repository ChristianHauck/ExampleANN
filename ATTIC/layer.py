#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#


# from multipledispatch import dispatch
#  Info about method overloading / dispatching: http://stackoverflow.com/questions/6434482/python-function-overloading
from functions import Functions
import numpy as np
import math


class Layer():

    def __init__(self, num_in, num_nodes, threshold=1, activation='tanh'):
        self.num_in = num_in
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.functions = Functions()
        self.init_activation(activation)


    def init_activation(self, activation):
        self.activation = self.functions.get_fu[activation]
        self.deriv_act = self.functions.get_deriv[activation]


    def init_values(self, v_in, m_weights):
        assert v_in.ndim == 1 and self.num_in == v_in.size
        self.in_nodes = v_in
        self.v_threshold = np.full(self.in_nodes, self.threshold)
        assert m_weights.ndim == 2 and m_weights.shape == (self.num_in, self.num_nodes)
        self.weight_matrix = m_weights
        return


    def init_weights(self, m_weights):
        assert m_weights.ndim == 2 and m_weights.shape == (self.num_nodes, self.num_in)
        self.weight_matrix = m_weights
        self.v_threshold = np.full(self.num_nodes, self.threshold)
        return


    def train(self):
        pass

    def learn_w_thr(self, M_Input, v_target, learning_rate=0.1):
        max_loops = 1000

        for i in range(self.weight_matrix.shape[0]):
            loop_nr = 0
            while True:
                delta_sum = 0
                error_sum = 0
                for j in range(M_Input.shape[0]):
                    actual = np.dot(self.weight_matrix[i], M_Input[j]) # - self.threshold

                    Functions.theta = self.threshold    # store threshold as Function-property for sign-functions
                    y = self.activation(actual)
                    err = v_target[j] - y # self.activation(actual, self.threshold)

                    # adjust threshold
                    self.threshold = self.threshold - learning_rate * err

                    # adjust weights
                    delta = (learning_rate * err * M_Input[j])  # ERROR! DerivActivation Function missing!!!
                    self.weight_matrix[i] = self.weight_matrix[i] + delta

                    #
                    # delta_sum = Functions.euclid_norm(delta_sum + err)
                    error_sum = error_sum + (err * err)
                    loop_nr += 1

                error_sum = math.sqrt(error_sum)
                if error_sum < 0.0001 or loop_nr > max_loops:
                    print("learning loops: ", loop_nr)
                    break  # while

        return


    def learn_wo_thr(self, M_Input, v_target, learning_rate=0.1):
        max_loops = 2000

        for i in range(self.weight_matrix.shape[0]):
            loop_nr = 0
            while True:
                error_sum = 0
                for j in range(M_Input.shape[0]):
                    actual = np.dot(self.weight_matrix[i], M_Input[j])
                    y = self.activation(actual)   # self.activation(actual)
                    err = v_target[j] - y
                    # adjust weights
                    delta = learning_rate * err * self.deriv_act(actual) * M_Input[j]
                    self.weight_matrix[i] = self.weight_matrix[i] + delta
                    #
                    error_sum = error_sum + (err * err)
                weights = self.weight_matrix[i]
                loop_nr += 1

                # error_sum = math.sqrt(error_sum)
                if error_sum < 0.0001 or loop_nr >= max_loops:
                    print("learning loops: ", loop_nr)
                    break  # while

        return


    def show(self, out_str, M_in, v_desired_out):
        print(out_str)
        # M_in = np.array([[-1, 1, 1]])
        print("weigt matrix: ", self.weight_matrix, " threshold: ", self.threshold)
        for i in range(M_in.shape[0]):
            result = self.run_wo_thr(M_in[i])
            print("input ", M_in[i], "desired output ", v_desired_out[i], " output ", result)


    def run_empty(self):
        self.v_nodes = np.dot(self.in_nodes, self.weight_matrix)
        res = Functions.sigmoid(self.v_nodes)
        return res


    def run_w_thr(self, v_in):
        assert v_in.ndim == 1 and self.num_in == v_in.shape[0]
        self.in_nodes = v_in
        self.v_nodes = np.dot(self.weight_matrix, self.in_nodes)
        self.v_nodes = self.v_nodes - self.v_threshold
        res = self.activation(self.v_nodes)
        return res


    def run_wo_thr(self, v_in):
        assert v_in.ndim == 1 and self.num_in == v_in.shape[0]
        self.in_nodes = v_in
        self.v_nodes = np.dot(self.weight_matrix, self.in_nodes)
        res = self.activation(self.v_nodes)
        return res


if __name__ == '__main__':
    # Main entry point for tests

    # Single-Layer-Perceptron
    layer = Layer(num_in=4, num_nodes=2)
    v_in = np.array([0, 0, 0, 1])
    m_weights = np.random.rand(4, 2)
    layer.init_values(v_in, m_weights, 1)
    layer.run()