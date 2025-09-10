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

class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
class NestedUNetHeTrans(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, init_type='kaiming'):
        super(NestedUNetHeTrans, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusting ConvTranspose2d layers for correct channel dimensions
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)

        nn.init.kaiming_normal_(self.up1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up4.weight, mode='fan_in', nonlinearity='relu')

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[0], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[1], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[2], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[3], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[0], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[1], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[2], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[0], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[1], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[0], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()


        init_weights(self, init_type)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3)], 1))

        output = self.final(x0_4)
        output = self.sigmoid(output)
        return output

class NestedUnet_DeepSup(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, init_type='kaiming'):
        super(NestedUnet_DeepSup, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adjusting ConvTranspose2d layers for correct channel dimensions
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)

        nn.init.kaiming_normal_(self.up1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up4.weight, mode='fan_in', nonlinearity='relu')

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[0], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[1], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[2], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[3], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[0], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[1], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[2], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[0], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[1], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[0], filters[0], filters[0])

        # Final output layer
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        
        # Deep supervision layers
        self.ds0_1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.ds0_2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.ds0_3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.ds0_4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

        init_weights(self, init_type)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1(x1_3)], 1))

        # Final output
        final_output = self.sigmoid(self.final(x0_4))
        
        # Deep supervision outputs
        ds0_1_output = self.sigmoid(self.ds0_1(x0_1))
        ds0_2_output = self.sigmoid(self.ds0_2(x0_2))
        ds0_3_output = self.sigmoid(self.ds0_3(x0_3))
        ds0_4_output = self.sigmoid(self.ds0_4(x0_4))

        return final_output, ds0_1_output, ds0_2_output, ds0_3_output, ds0_4_output
