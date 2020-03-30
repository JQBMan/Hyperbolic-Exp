##################
# tanh function
##################
import torch


def tanh(x, clamp=5):
  return x.clamp(-clamp, clamp).tanh()
##################
# artanh function
##################
def artanh(x):
  return Artanh.apply(x)
###################
#class Artanh
###################
class Artanh(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    x_dtype = x.dtype
    x = x.double()
    x = x.clamp(-1 + 1e-5, 1 - 1e-5)
    ctx.save_for_backward(x)
    z = x.double()
    temp = torch.log_(1 + z).sub_(torch.log_(1 - z))
    res = (temp).mul_(0.5).to(x_dtype)
    return res

  @staticmethod
  def backward(ctx, grad_output):
    input, = ctx.saved_tensors
    return grad_output / (1 - input ** 2)