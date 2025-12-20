import torch
from torch import nn
from torch.autograd import Function
import moe_ffn_ext  # the compiled pytorch cuda extension


class MoEExpertFFNFunction(Function):
    '''
    custom autograd func to wrap cuda FFN implmentation
    connects pytorch autograd to custom cuda kernels
    represents a single expert ffn
    '''
    @staticmethod
    def forward(ctx, x, W1, b1, W2, b2):
        '''
         inputs: x: [N, Din], W1: [Din, Dhid], b1: [Dhid], W2: [Dhid, Dout], b2: [Dout]
         outputs: y: [N, Dout]
        '''
        # Ensure tensors is contiguous since it would lead to incorrect mem reads
        x_  = x.contiguous()
        W1_ = W1.contiguous()
        b1_ = b1.contiguous()
        W2_ = W2.contiguous()
        b2_ = b2.contiguous()
        #calls the kernel
        y, h, a = moe_ffn_ext.forward(x_, W1_, b1_, W2_, b2_)

        # Saves the tensors for backward pass
        ctx.save_for_backward(x_, h, a, W1_, W2_)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        inputs: grad_y: [N, Dout]
        outputs: (dx, dW1, db1, dW2, db2)
        """
        #get saved tensors and ensure contigous before cuda opertions 
        x, h, a, W1, W2 = ctx.saved_tensors
        dy = grad_y.contiguous()
        #calls kernel
        dx, dW1, db1, dW2, db2 = moe_ffn_ext.backward(
            dy, x, h, a, W1, W2
        )

        # Gradients wrt inputs in same order as forward:
        return dx, dW1, db1, dW2, db2


def moe_expert_ffn(x, W1, b1, W2, b2):
    return MoEExpertFFNFunction.apply(x, W1, b1, W2, b2)