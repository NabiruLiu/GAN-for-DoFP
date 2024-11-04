import numpy as np

def normalize(data, lower, upper):
    mx = np.max(data)
    mn = np.min(data)
    if mx==mn:
        norm_data = np.zeros(data.shape)
    else:  
        norm_data = (upper-lower)*(data - mn) / (mx - mn) + lower
    return norm_data

'''
Calculate the AoP
'''
def aop(x_0, x_45, x_90, x_135, normalization = True):
    AoP = 0.5 * np.arctan2((x_45 - x_135), (x_0 - x_90)) # range in [-pi/2, pi/2]
    
    if normalization:
        AoP = (AoP + np.pi / 2) / np.pi

    return AoP

'''
Calculate the DoLP
'''
def dolp(i0, i45, i90, i135, normalization = False):
    s0 = 0.5*(i0 + i45 + i90 + i135)   
    DoLP = np.sqrt(np.square(i0-i90) + np.square(i45-i135))/(s0+1e-8)
    DoLP[np.where(s0==0)] = 0
    if normalization:
        DoLP = normalize(DoLP,0,1)
    
    return DoLP
