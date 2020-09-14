import torch
import math

def abs(input, inplace):
    return F_Abs.apply(input, inplace)


def lowerbound(input, min):
    return F_Lowerbound.apply(input, min)


def round(input):
    return F_Round.apply(input)


def erf(input):
    return F_Erf.apply(input)


class F_Abs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, inplace):
        if inplace:
            return input.abs_()
        else:
            ctx.save_for_backward(input)
            return input.abs()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] *= -1
        return grad_input, None


class F_Lowerbound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min):
        ctx.min = min
        ctx.input = input
        # ctx.save_for_backward(input)
        bound = input.new(input.size()).fill_(min)
        output = input.clone()
        output[input < bound] = min
        return output

    @staticmethod
    def backward(ctx, grad_output):
        min = ctx.min
        input = ctx.input
        # input = ctx.saved_tensors
        bound = input.new(input.size()).fill_(min)
        grad_input = grad_output.clone()
        mask = input.new(input.size()).fill_(0)
        mask[input >= bound] = 1
        mask[grad_input < 0] = 1
        grad_input[mask == 0] = 0
        return grad_input, None


class F_Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class F_Erf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.erf(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.data *= (torch.exp(-input.pow(2)) * (2/(math.pi ** 0.5)))
        return grad_input