import torch.optim as optim

class Optimizer:
  """
  Wrapper around torch.optim + learning rate
  """
  def __init__(self, params, method, base_lr, weight_decay, **kwargs):
    self.method = method
    self.base_lr = base_lr

    if self.method == 'sgd':
      self.learner = optim.SGD(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'adam':
      self.learner = optim.Adam(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'rmsprop':
      self.learner = optim.RMSprop(params, lr=self.base_lr,
        weight_decay=weight_decay, **kwargs)
    elif self.method == 'Adadelta':
      self.learner = optim.Adadelta(params, lr=self.base_lr, weight_decay=weight_decay)
    else:
      print("No such optimizier!")

  def adjust_lr(self, epoch): # for SGD
    if self.method != 'sgd':
      return self.base_lr

    decay_factor = 1
    for s in self.lr_stepvalues:
      if epoch < s:
        break
      decay_factor *= self.lr_decay

    lr = self.base_lr * decay_factor

    for param_group in self.learner.param_groups:
      param_group['lr'] = lr

    return lr

  def mult_lr(self, f):
    for param_group in self.learner.param_groups:
      param_group['lr'] *= f