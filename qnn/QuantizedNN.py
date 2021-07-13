import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np


class Quantization:
    def __init__(self, method):
        self.method = method
    def applyQuantization(self, input):
        return self.method(input)
        

class Quantize(Function):
    @staticmethod
    def forward(ctx, input, quantization):
        """[summary]

        Args:
            ctx (any): ctx is a context object that can be used to stash information
             for backward computation.You can cache arbitrary and objects 
             for use in the backward pass using the ctx.save_for_backward method
             example: ctx.save_for_backward(input, weight, bias)

            input (any): [description]
            quantization ([type]): [description]

        Returns:
            [type]: [description]
        """

        output = input.clone().detach()
        output = quantization.applyQuantization(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """gradient formula

        Args:
            ctx (any): ctx is a context object that can be used to access stashed information
             for backward computation saved in the forward method. 
            
            grad_output (any): gradient of the loss with respect to the output,
            and we need to compute the gradient of the loss with respect to the input.

        Returns:
            grad_input (any): gradient of the loss with respect to the input.
            since there are two inputs in the forward method, there should be two returns
            but the associated with quantization input argument is none 
        """
        grad_input = grad_output.clone()
        return grad_input, None  # gradients of input and quantization(none) in forward function

quantize = Quantize.apply # aliase

class QuantizedActivation(nn.Module):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedActivation"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        super(QuantizedActivation, self).__init__(*args, **kwargs)

    def forward(self, input):
        output = quantize(input, self.quantization)
        return output


class QuantizedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedLinear"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        super(QuantizedLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = quantize(self.weight, self.quantization)
            output = F.linear(input, quantized_weight)
            return output
        else:
            quantized_weight = quantize(self.weight, self.quantization)
            quantized_bias = quantize(self.bias, self.quantization)
            return F.linear(input, quantized_weight, quantized_bias)


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.name = "QuantizedConv2d"
        self.layerNR = kwargs.pop('layerNr', None)
        self.quantization = kwargs.pop('quantization', None)
        super(QuantizedConv2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        if self.bias is None:
            quantized_weight = quantize(self.weight, self.quantization)
            output = F.conv2d(input, quantized_weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
        else:
            quantized_weight = quantize(self.weight, self.quantization)
            quantized_bias = quantize(self.bias, self.quantization)
            output = F.conv2d(input, quantized_weight, quantized_bias, self.stride,
                              self.padding, self.dilation, self.groups)
            return output
