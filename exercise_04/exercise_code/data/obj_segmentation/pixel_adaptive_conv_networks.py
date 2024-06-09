from numbers import Number
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def nd2col(
    input_nd,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    # transposed=False,
):
    """
    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C, *kernel_size, *L_{out})` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)` for non-transposed
          :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1 + output_padding` for transposed
    """
    n_dims = len(input_nd.shape[2:])
    kernel_size = (
        (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    )
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (
        (output_padding,) * n_dims
        if isinstance(output_padding, Number)
        else output_padding
    )
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    # NOTE: make a possible task to implement the correct output size of the convolution operation
    out_sz = tuple(
        [
            ((i + 2 * p - d * (k - 1) - 1) // s + 1)
            for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)
        ]
    )

    output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
    out_shape = (bs, nch) + tuple(kernel_size) + out_sz
    output = output.view(*out_shape).contiguous()
    return output


def packernel2d(
    input: torch.Tensor,
    kernel_size=0,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)

    bs, k_ch, in_h, in_w = input.shape

    x = nd2col(input, kernel_size, stride, padding, output_padding, dilation) 
    x = x.mean(dim=(0, 1), keepdim=True) 
    # print("==: ", x.shape)

    center_pixel = (kernel_size[0] // 2, kernel_size[1] // 2)
    center_feature = x[:, :, center_pixel[0], center_pixel[1]]

    out_h = (in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
    # x = x.view(bs, k_ch, kernel_size[0], kernel_size[1], out_h, out_w)
    feat_0 = x.contiguous()[:, :, center_pixel[0]:center_pixel[0] + 1, center_pixel[1]:center_pixel[1] + 1, :, :]
    # feat_0 = torch.ones_like(feat_0)
    center_feature.fill_(1)
    
    diff_sq = (x - feat_0).pow(2).sum(dim=1, keepdim=True)
    output = torch.exp(-0.5 * diff_sq)

    output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()

    return output


def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)



    bs, k_ch, in_h, in_w = input.shape
    x = nd2col(input, kernel_size, stride, padding, dilation)
    in_mul_k = x.view(bs, k_ch, *kernel.shape[2:]) * kernel
    output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k, weight))

    return output
