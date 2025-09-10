import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p

class attention_gate(nn.Module):
    def __init__(self, g_c, s_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(g_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(s_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)  # This operates on decoder output
        Ws = self.Ws(s)  # This operates on skip connection
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, convTranspose=True):
        super().__init__()

        if convTranspose:
            self.up = nn.ConvTranspose2d(in_channels=in_c[0], out_channels=out_c, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.ag = attention_gate(g_c=out_c, s_c=in_c[1], out_c=out_c)
        
        # Ensure concatenation channels are correctly matched
        self.c1 = conv_block(out_c + in_c[1], out_c)  

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)  # Concatenation
        x = self.c1(x)
        return x

class attention_unet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.e1 = encoder_block(in_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        self.b1 = conv_block(512, 1024)

        self.d1 = decoder_block([1024, 512], 512)
        self.d2 = decoder_block([512, 256], 256)
        self.d3 = decoder_block([256, 128], 128)
        self.d4 = decoder_block([128, 64], 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        init_weights(self)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)  # FIXED: use e4 instead of e3

        b1 = self.b1(p4)
        
        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)  # Also fixed: use d2 here, not d1
        d3 = self.d3(d2, s2)  # Fixed: use d3 here
        d4 = self.d4(d3, s1)  # Fixed: use d4 here

        output = self.output(d4)
        return torch.sigmoid(output)

if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    # model = attention_unet()
    # output = model(x)
    # print(output.shape)
