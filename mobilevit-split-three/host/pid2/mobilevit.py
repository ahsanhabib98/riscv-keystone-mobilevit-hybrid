import os
import errno
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=2):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_res_connect = (stride == 1 and inp == oup)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, heads=4, dim_head=8, mlp_dim=mlp_dim, dropout=dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(channel*2, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        B, D, H, W = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(
            x,
            'b (ph pw) (h w) d -> b d (h ph) (w pw)',
            h=H//self.ph, w=W//self.pw, ph=self.ph, pw=self.pw
        )
        x = self.conv3(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv4(x)
        return x

class MobileViTMiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.mv2  = MV2Block(inp=16, oup=16, stride=1, expansion=2)
        self.mvit = MobileViTBlock(dim=64, depth=1, channel=16,
                                   kernel_size=3, patch_size=(2,2), mlp_dim=128)
        self.conv2 = conv_1x1_bn(16, 64)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.mv2(x)               
        x = self.mvit(x)              
        x = self.conv2(x)             
        x = self.pool(x)              
        B, C, H, W = x.shape
        x = x.view(B, -1)             
        return x

def print_time(message):
    now = datetime.now()
    ts  = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"Time: {ts} : {message}")
    
def main():

    INPUT_FILE  = '/tmp/input.bin'
    OUTPUT_FILE = '/tmp/output.bin'

    data = np.fromfile(INPUT_FILE, dtype=np.float32)
    data = data.reshape(1, 16, 56, 112)

    print('MobileViTMiddle')
    model = MobileViTMiddle()       
    x = torch.from_numpy(data)
    out = model(x)                  

    out_np = out.squeeze(0).detach().numpy().astype(np.float32).ravel()
    out_np.tofile(OUTPUT_FILE)

if __name__ == '__main__':
    main()
