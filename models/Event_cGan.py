import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from .BasicModule import BasicModule

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class D(BasicModule):
    def __init__(self, nc, nf):
        super(D, self).__init__()

        main = nn.Sequential()
        # 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        main.add_module('%s_conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

        # 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

        # 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
        main.add_module('%s_bn' % name, nn.BatchNorm2d(nf*2))


        # # 31
        layer_idx += 1
        name = 'layer%d' % layer_idx
        nf = nf * 2
        main.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        main.add_module('%s_conv' % name, nn.Conv2d(nf, 1, 3, 1, 1, bias=False))

        main2=nn.Sequential()
        main2.add_module('%fc',nn.Linear(31*31*1,1))
        main2.add_module('%sigmoid' , nn.Sigmoid())
        self.main1 = main
        self.main2= main2

    def forward(self, x):
        output = self.main1(x)
        output=output.view(output.size()[0],-1)
        output=self.main2(output)
        return output


class G(BasicModule):
    def __init__(self, input_nc, output_nc, nf):
        super(G, self).__init__()

        # 1 input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # 2 input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
        # 3 input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
        # 4 input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # 5 input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # 6 input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        #7  input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # 8 input is 2 x  2
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # 8 input is 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8
        dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

        #import pdb; pdb.set_trace()
        # 7 input is 2
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        # 6 input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        #5  input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # 4 input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*8*2
        dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
        # 3 input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*4*2
        dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
        #2  input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf*2*2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        #1 input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf*2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        out1 = self.layer1(x)
        #print('out1.size=',out1.size())
        out2 = self.layer2(out1)
        #print('out2.size=', out2.size())
        out3 = self.layer3(out2)
        #print('out3.size=', out3.size())
        out4 = self.layer4(out3)
        #print('out4.size=', out4.size())
        out5 = self.layer5(out4)
        #print('out5.size=', out5.size())
        out6 = self.layer6(out5)
        #print('out6.size=', out6.size())
        out7 = self.layer7(out6)
        #print('out7.size=', out7.size())
        out8 = self.layer8(out7)
        #print('out8.size=', out8.size())
        dout8 = self.dlayer8(out8)
        #print('dout8.size=', dout8.size())
        dout8_out7 = torch.cat([dout8, out7], 1)
        #print('dout8_cat.size=', dout8_out7.size())
        dout7 = self.dlayer7(dout8_out7)
        #print('dout7.size=', dout7.size())
        dout7_out6 = torch.cat([dout7, out6], 1)
        #print('dout7_cat.size=', dout7_out6.size())
        dout6 = self.dlayer6(dout7_out6)
        #print('dout6.size=', dout6.size())
        dout6_out5 = torch.cat([dout6, out5], 1)
        #print('dout6_cat.size=', dout6_out5.size())
        dout5 = self.dlayer5(dout6_out5)
        #print('dout5.size=', dout5.size())
        dout5_out4 = torch.cat([dout5, out4], 1)
        #print('dout5_cat.size=', dout5_out4.size())
        dout4 = self.dlayer4(dout5_out4)
        #print('dout4.size=', dout4.size())
        dout4_out3 = torch.cat([dout4, out3], 1)
        #print('dout4_cat.size=', dout4_out3.size())
        dout3 = self.dlayer3(dout4_out3)
        #print('dout3.size=', dout3.size())
        dout3_out2 = torch.cat([dout3, out2], 1)
        #print('dout3_cat.size=', dout3_out2.size())
        dout2 = self.dlayer2(dout3_out2)
        #print('dout2.size=', dout2.size())
        dout2_out1 = torch.cat([dout2, out1], 1)
        #print('dout2_cat.size=', dout2_out1.size())
        dout1 = self.dlayer1(dout2_out1)
        #print('dout1.size=', dout1.size())
        return dout1
