import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

# Define the Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# Define the Attention ResUNet
class AttentionResUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionResUNet, self).__init__()
        filters = [64, 128, 256, 512, 1024]  # Added 1024 filter level

        # Encoder path
        self.conv1 = ResidualBlock(in_channels, filters[0])
        self.conv2 = ResidualBlock(filters[0], filters[1])
        self.conv3 = ResidualBlock(filters[1], filters[2])
        self.conv4 = ResidualBlock(filters[2], filters[3])
        self.conv5 = ResidualBlock(filters[3], filters[4])  # New 1024 filter layer

        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder path with Attention
        self.upconv4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.conv4d = ResidualBlock(filters[4], filters[3])

        self.upconv3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.conv3d = ResidualBlock(filters[3], filters[2])

        self.upconv2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.conv2d = ResidualBlock(filters[2], filters[1])

        self.upconv1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=filters[0], F_l=filters[0], F_int=32)
        self.conv1d = ResidualBlock(filters[1], filters[0])

        # Output layer
        self.output_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Apply He initialization to the entire model
        init_weights(self)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        c4 = self.conv4(p3)
        p4 = self.pool(c4)
        c5 = self.conv5(p4)  # 1024 Filters

        # Decoder with Attention
        u4 = self.upconv4(c5)
        x4 = self.att4(g=u4, x=c4)
        c4d = self.conv4d(torch.cat([u4, x4], dim=1))

        u3 = self.upconv3(c4d)
        x3 = self.att3(g=u3, x=c3)
        c3d = self.conv3d(torch.cat([u3, x3], dim=1))

        u2 = self.upconv2(c3d)
        x2 = self.att2(g=u2, x=c2)
        c2d = self.conv2d(torch.cat([u2, x2], dim=1))

        u1 = self.upconv1(c2d)
        x1 = self.att1(g=u1, x=c1)
        c1d = self.conv1d(torch.cat([u1, x1], dim=1))

        # Output with sigmoid activation
        output = self.output_conv(c1d)
        output = self.sigmoid(output)
        return output

# Example usage
if __name__ == "__main__":
    model = AttentionResUNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)
