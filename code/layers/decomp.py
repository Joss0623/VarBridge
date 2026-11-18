import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

class SMA(nn.Module):
    """
    Simple Moving Average (SMA) block to highlight the trend of time series
    (Standard implementation from Autoformer/PatchTST)
    """
    def __init__(self, kernel_size=25, stride=1):
        super(SMA, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    
        x_padded = torch.cat([front, x, end], dim=1)
   
        x_avg = self.avg(x_padded.permute(0, 2, 1))
        
        return x_avg.permute(0, 2, 1)
    
class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta, kernel_size):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'sma':
            self.ma = SMA(kernel_size)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average