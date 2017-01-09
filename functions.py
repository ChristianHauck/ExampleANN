#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#
# References:
#   Hertz/Krogh/Palmer, Introduction to the Theory of Neural Computation, Addison-Wesley, 1991

import numpy as np
import math


class Functions():
    def __init__(self):
        pass


    def test(self):
        pass


    def sigmoid(v_in):
        ones = np.ones_like(v_in)
        zeros = np.zeros_like(v_in)
        minus_x = zeros - v_in
        cx = minus_x / 1    # 1 is the "temperature-parameter"
        result = ones / (ones + np.exp(cx))
        return result


    def deriv_sigmoid(v_in):
        # d/dx s(x) = s(x)(1-s(x))
        ones = np.ones_like(v_in)
        # result = sigmoid(v_in) * (ones - sigmoid(v_in))
        zeros = np.zeros_like(v_in)
        minus_x = zeros - v_in
        cx = minus_x / 1
        result = (ones / (ones + np.exp(cx))) * (ones - (ones / (ones + np.exp(cx))))
        return result


    def tanh(v_in):
        v_out = np.tanh(v_in)
        return v_out


    def deriv_tanh(v_in):
        # d/dx tanh(x) = 1-tanh^2(x)
        ones = np.ones_like(v_in)
        v_out = ones - (np.tanh(v_in) * np.tanh(v_in))
        return v_out


    def id_fu(self, v_in):
        return v_in


    def deriv_id_fu(self, v_in):
        if type(v_in) is list:
            ones = np.ones_like(v_in)
            return ones
        return 1


    @staticmethod
    def signum(v_in):
        v_out = np.sign(v_in)
        return v_out


    theta = 0.0


    @staticmethod
    def threshold(v_in):
        if type(v_in) is list:
            v_out = np.copy(v_in)
            for i in range(v_in.shape[0]):
                if v_in[i] >= Functions.theta:
                    v_out[i] = 1
                else:
                    v_out[i] = 0
            return v_out
        else:
            if v_in >= Functions.theta:
                return 1
            else:
                return 0


    @staticmethod
    def depricated_int_threshold(x):
        if x >= Functions.theta :
            return 1
        else:
            return 0



    #
    #   ---===### Function Directories ###===---
    #
    get_fu = {'sigmoid': sigmoid, 'tanh' : tanh, 'lin' : id_fu, 'threshold' : threshold}
    get_deriv = {'sigmoid': deriv_sigmoid, 'tanh': deriv_tanh, 'lin': id_fu}



    #
    #   ---===### HELPER FUNCTIONS ###===---
    #
    @staticmethod
    def euclid_norm(vector):
        sum = 0
        for i in range(vector.size):
            sum += vector[i] * vector[i]
        result = math.sqrt(sum)
        return result


    @staticmethod
    def squared_error(v_target, v_observed):
        assert v_target.size == v_observed.size
        sum = 0
        for i in range(v_target.size):
            err = v_target[i] - v_observed[i]
            sum = sum + np.dot(err, err)
        result = 1.0 / v_target.size * sum
        return result



if __name__ == '__main__':
    # Main entry point for tests
    func = Functions()
    func.test()