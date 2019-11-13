'''
author: longxin
date: 2019-11-12
description: the basic example of neural networks
changeversion: 0.0
changedescripiton:

'''

'''
the example of full connected 
usage example:
fl = FullconnectedLayer((128, 128), 100)
tmp  = fl.compute(x)
'''


class FullconnectedLayer:

    def __init__(self, x_shape, neural_num):
        '''
        init function
        :param x_shape: the shape of init
        :param neural_num: the count of neural
        '''
        import numpy as np

        self.W = np.random.random(size=[neural_num] + list(x_shape))
        self.bias = np.random.random(size=[neural_num])

        self.neural_num = neural_num

    def compute(self,x):
        import numpy as np
        output = []
        for j in range(self.neural_num):
            output.append(np.sum(self.W[j]*x) + self.bias[j])
        return output

'''
the example of convenlutaion network
usage example:
cl = ConvenlutionLayer((3, 3), 1, 1, (128, 128), 10)
cl.compute(np.random.random((128, 128, 1)))
'''


class ConvenlutionLayer:

    def __init__(self, featuremap_shape, strides, input_channel, input_shape, featuremap_count):
        '''
        init the convenlution layer
        :param featuremap_shape: 过滤器大小
        :param strides: 步长
        :param input_channel: 连接的领域 e.t. 黑白： [1]  rgb:[3]  采样形状：[5, 5]
        :param featuremap_count: 过滤器数量
        :param input_shape: 输入形状 二维 [128, 128]
        '''
        import numpy as np

        # 初始化参数
        self.W = np.random.random(size=list(featuremap_shape) + [input_channel] + [featuremap_count])
        self.bias = np.random.random(size=list(featuremap_shape) + [featuremap_count])

        # 初始化输出特征
        self.outputshape_w = int(input_shape[0]/strides)
        self.outputshape_h = int(input_shape[1]/strides)
        self.featuremap = np.zeros(shape=(featuremap_count, self.outputshape_w, self.outputshape_h))

        # 初始化卷积相关参数
        self.featuremap_count = featuremap_count
        self.featuremap_shape = featuremap_shape

        self.input_shape = input_shape
        self.input_channel = input_channel

    def compute(self, x):
        '''
        computer the convelution layer
        :param x: input data  shape is x_shape
        :return: the convelution output
        '''
        import numpy as np

        # 对输入进行padding 采用 zero padding 的形式
        padding_example = np.zeros(shape=[self.input_shape[0] + self.featuremap_shape[0] - 1, self.input_shape[1] +
                                          self.featuremap_shape[1] - 1, self.input_channel])
        padding_example[:-(self.featuremap_shape[0]-1), (self.featuremap_shape[1]-1):, :] = x

        for i in range(self.featuremap_count):
            for ow in range(self.outputshape_w):
                for oh in range(self.outputshape_h):
                    for kw in range(self.featuremap_shape[0]):
                        for kh in range(self.featuremap_shape[1]):
                            self.featuremap[i, ow, oh] += padding_example[ow+kw, oh+kh, :] * self.W[kw, kh, :, i] \
                                                          + self.bias[kw, kh, i]

        return self.featuremap
