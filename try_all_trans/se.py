import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_

class DSConv(nn.Module):
    """Depthwise Separable Convolution for 1D/2D/3D inputs.
    Use 3*3 and 1*1 Conv to reduce the channel of Q, K, V to out_ch.

    Parameters:
        in_ch(int): input channel

        out_ch(int): output channel

        conv_type(str): convolution type, can selelct '1d', '2d' or '3d'.

        stride(int): stride of the convolution

        kernel_size(int): kernel size of the convolution

        padding(int): padding of the convolution

        bias(bool): whether to use bias in the convolution

        xavier_init(float): the gain of the xavier initialization
    """

    def __init__(self, in_ch, out_ch, conv_type, stride=1, kernel_size=3, padding=1, bias=True, xavier_init=1e-2):
        super().__init__()

        if conv_type == '1d':
            self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel_size,
                                       padding=padding, groups=in_ch, bias=bias, stride=stride)
            self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
        elif conv_type == '2d':
            self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=bias, stride=stride)
            self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        elif conv_type == '3d':
            self.depthwise = nn.Conv3d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=bias, stride=stride)
            self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=bias)
        else:
            raise ValueError(f'conv_type should be 1d, 2d or 3d, but got {conv_type}.')
        
        self.xavier_init = xavier_init
        self._reset_parameters()

    def forward(self, x):
        """
        Parameters:
            x(tensor): input tensor, shape is 3D or 4D.

        Returns:
            tensor: output tensor, shape is 3D or 4D, same as input.
        """
        out = self.depthwise(x)  
        out = self.pointwise(out)  

        return out

    def _reset_parameters(self):
        for layer in [self.depthwise, self.pointwise]:
            for param in layer.parameters():
                if param.ndim > 1:
                    xavier_uniform_(param, gain=self.xavier_init)
                else:
                    constant_(param, 0)
x = torch.randn(2, 1, 3, 4, 1)

se = DSConv(in_ch=1, out_ch=6, conv_type='3d')
y = se(x)
print(y.shape)

