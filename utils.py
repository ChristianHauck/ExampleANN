#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017, Christian Hauck
#
# Author: Christian Hauck, <christian_hauck@yahoo.com>
#

import numpy as np
import struct


class Utils:

    mnist_train_data   = "train-images-idx3-ubyte"
    mnist_train_labels = "train-labels-idx3-ubyte"
    mnist_test_data    = "t10k-images-idx3-ubyte"
    mnist_test_labels  = "t10k-labels-idx3-ubyte"

    @classmethod
    def load_mnist_data(cls, file_name, max_items=-1):
        f = open('../data/' + file_name, 'r')
        magic = f.read(4)  # ignore
        num_items = struct.unpack('>i', f.read(4))[0]
        if max_items >= 0:
            num_items = max_items
        rows = struct.unpack('>i', f.read(4))[0]
        columns = struct.unpack('>i', f.read(4))[0]
        #
        data = np.fromfile(f, dtype=np.ubyte)
        f.close()
        #
        data = data.reshape((num_items, rows*columns))
        data = data / 255.0  # keep unaugmented assignment because I do not trust numpy so far ;)
        #
        return [num_items, rows, columns, data]

