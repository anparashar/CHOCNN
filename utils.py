import numpy as np
from copy import deepcopy

try:
    # Python 2 module
    from itertools import izip_longest as zip_longest
except ImportError:
    # Python 3 module
    from itertools import zip_longest


def add_convinit(layers, out_c, conv_kernel):
    layers.append({"type": "conv", "ou_c": out_c, "kernel": conv_kernel})
    return layers


def add_poolinit(layers):
  layers.append({"type": "max_pool", "ou_c": -1, "kernel": 3})
  return layers


def add_conv(layers, max_out_ch, conv_kernel):
    out_channel = np.random.randint(3, max_out_ch)
    conv_kernel = np.random.randint(3, conv_kernel)
    layers.append({"type": "conv", "ou_c": out_channel, "kernel": conv_kernel})
    return layers


def add_fc(layers, max_fc_neurons):
    layers.append({"type": "fc", "ou_c": max_fc_neurons, "kernel": -1})
    return layers
        

# def add_pool(layers, pool_kernel):
#         # pool_kernel = np.random.randint(3, pool_kernel)
#         # Add Max Pooling
#         layers.append({"type": "max_pool", "ou_c": -1, "kernel": 3})
#         return layers

def add_pool(layers, pool_kernel):
    
        random_pool = np.random.rand()
        
        if random_pool <= 0.5:
            # Add Max Pooling
            layers.append({"type": "max_pool", "ou_c": -1, "kernel": 3})
        else:
            layers.append({"type": "avg_pool", "ou_c": -1, "kernel": 3})
    
        return layers



def computeVelocity(gBest, pBest, p):
    # a1 = np.random.uniform(0.0 , 0.2)
    # a1 = np.random.rand()
    a1 = 0.5
    velocity = []
    for i in range(len(p)):
      if p[i]["type"]== "conv" :
        nkc = p[i]["ou_c"]
        nkp = pBest[i]["ou_c"]
        nkg = gBest[i]["ou_c"]
        M1 = (nkp + nkg)/2
        OUT_CH = round(nkc + a1*(M1 - nkc))

        abc = p[i]["kernel"]
        abp = pBest[i]["kernel"]
        abg = gBest[i]["kernel"]
        M2 = (abp + abg)/2
        KERNEL = round(abc + a1*(M2 - abc))
        
        velocity.append({"type": "conv", "ou_c": OUT_CH, "kernel": KERNEL})
      
    # print(velocity)
      if p[i]["type"]== "max_pool" and pBest[i]["type"]== "max_pool" and gBest[i]["type"]== "max_pool" :
        velocity.append(pBest[i])
      if p[i]["type"]== "avg_pool" and pBest[i]["type"]== "avg_pool" and gBest[i]["type"]== "avg_pool" :
        velocity.append(pBest[i])
      if p[i]["type"]== "max_pool" and pBest[i]["type"]== "max_pool" and gBest[i]["type"]== "avg_pool" : 
        velocity.append(p[i])
      if p[i]["type"]== "max_pool" and pBest[i]["type"]== "avg_pool" and gBest[i]["type"]== "max_pool" : 
        velocity.append(p[i])
      if p[i]["type"]== "avg_pool" and pBest[i]["type"]== "max_pool" and gBest[i]["type"]== "max_pool" : 
        velocity.append(pBest[i])
      if p[i]["type"]== "avg_pool" and pBest[i]["type"]== "avg_pool" and gBest[i]["type"]== "max_pool" : 
        velocity.append(p[i])
      if p[i]["type"]== "avg_pool" and pBest[i]["type"]== "max_pool" and gBest[i]["type"]== "avg_pool" : 
        velocity.append(p[i])
      if p[i]["type"]== "max_pool" and pBest[i]["type"]== "avg_pool" and gBest[i]["type"]== "avg_pool" : 
        velocity.append(pBest[i])
  ###############################################################3
      # if p[i]["type"]== "max_pool" :
      #   # abc = p[i]["kernel"]
      #   # abp = pBest[i]["kernel"]
      #   # abg = gBest[i]["kernel"]
      #   # M3 = (abp + abg)/2
      #   # kernel = round(abc + a1*(M3 - abc))
      #   # # print(kernel)
      #   # velocity.append({"type": "max_pool", "ou_c": -1, "kernel": kernel})
      #   velocity.append(p[i])
 ###############################################################################       
      if p[i]["type"]== "fc" :
        velocity.append(gBest[i])
        
    return velocity

def updateParticle(p, velocity):
    new_p = velocity
        
    return new_p
