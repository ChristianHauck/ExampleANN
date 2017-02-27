#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

from functions import Functions
import numpy as np


class Layer2:

    def __init__(self, num_in, num_out):
        self.num_in = num_in
        self.num_out = num_out
        self.num_weights = num_in * num_out
        # vectors and weight matrix needed for learning and running
        self.input = None
        self.output = None
        self.weights = None
        self.sum = None
        self.delta = None
        self.delta_w = None
        # vectors for interim values
        self.derived = None
        self.diff = None
        self.tmp_col = None
        #
        self.functions = Functions()


    def set_input(self, v_input):
        assert v_input.shape == (self.num_in,)


    def init_weights(self, m_weights):
        assert m_weights.shape == (self.num_out, self.num_in)
        self.weights = m_weights


    def show(self, out_str, inputs, v_desired_out):
        assert inputs.shape == v_desired_out.shape
        print(out_str)
        print("weight matrix: ", self.weights)
        for i in range(inputs.shape[0]):
            self.input = inputs[i]
            result = self.run()
            print("input ", inputs[i], "desired output ", v_desired_out[i], " output ", result)


    def run(self):
        self.sum = np.dot(self.input, self.weights)
        res = self.functions.get_fu['sigmoid'](self.sum)
        return res


if __name__ == '__main__':
    layer = Layer2(num_in=4, num_out=2)
    layer.set_input(np.array([0, 0, 0, 1]))
    layer.init_weights(np.random.rand(4, 2))
    layer.run()
