import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
from sklearn.metrics import confusion_matrix

def indices(pLabel, tLabel):
    # pass
    # pLabel shoulf after argmax
    confMat=confusion_matrix(tLabel, pLabel)
    # print(confMat)
    nc=np.sum(confMat, axis=1)
    tp=np.diagonal(confMat)
    tpr=tp/nc
    acsa=np.mean(tpr)
    gm=np.prod(tpr)**(1/confMat.shape[0])
    acc=np.sum(tp)/np.sum(nc)
    return acsa, gm, tpr, confMat, acc


def pairwise_distance(z, means):
        z = z.double()
        means = means.double()
        dist1 = (z**2).sum(dim=1).unsqueeze(1).expand(-1, means.shape[0]) #MB x C
        dist2 = (means**2).sum(dim=1).unsqueeze(0).expand(z.shape[0], -1) #MB x C
        dist3 = torch.mm(z, torch.transpose(means, 0, 1)) #MB x C
        return (dist1 + dist2 + - 2*dist3).float()

log_2_pi = math.log(2*math.pi)


def log_normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + log_2_pi + torch.pow(x - mean, 2) / torch.exp( log_var ) )
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)

def log_normal_diag_vectorized(x, mean, log_var):
    log_var_sqrt = log_var.mul(0.5).exp_()
    pair_dist = pairwise_distance(x/log_var_sqrt, mean/log_var_sqrt)
    log_normal = -0.5 * torch.sum(log_var+log_2_pi, dim=1) - 0.5*pair_dist
    return log_normal, pair_dist

def log_bernoulli(x, mean, average=False, dim=None):
    min_epsilon = 1e-5
    max_epsilon = 1.-1e-5
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)

    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)

class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights=['weight']):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

def he_init(m):
    s =  np.sqrt( 2. / m.in_features )
    m.weight.data.normal_(0, s)