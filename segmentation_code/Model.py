import torch.nn as nn
import torch


# class ResUnetPlusPlus(nn.Module):
#     def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
#         super(ResUnetPlusPlus, self).__init__()

#         self.input_layer = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
#                                          nn.BatchNorm2d(filters[0]),
#                                          nn.ReLU(),
#                                          nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
#                                         )
        
#         self.input_skip        = nn.Sequential(nn.Conv2d(channel, filters[0], kernel_size=3, padding=1))

#         self.squeeze_excite1   = Squeeze_Excite_Block(filters[0])
#         self.squeeze_excite2   = Squeeze_Excite_Block(filters[1])
#         self.squeeze_excite3   = Squeeze_Excite_Block(filters[2])

#         self.residual_conv1    = ResidualConv(filters[0], filters[1], 2, 1)
#         self.residual_conv2    = ResidualConv(filters[1], filters[2], 2, 1)
#         self.residual_conv3    = ResidualConv(filters[2], filters[3], 2, 1)

#         self.aspp_bridge       = ASPP(filters[3], filters[4])

#         self.attn1             = AttentionBlock(filters[2], filters[4], filters[4])
#         self.attn2             = AttentionBlock(filters[1], filters[3], filters[3])
#         self.attn3             = AttentionBlock(filters[0], filters[2], filters[2])

#         self.upsample1         = Upsample_(2)
#         self.upsample2         = Upsample_(2)
#         self.upsample3         = Upsample_(2)

#         self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)
#         self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)
#         self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

#         self.aspp_out          = ASPP(filters[1], filters[0])
#         self.output_layer      = nn.Sequential(nn.Conv2d(filters[0], 1, 1), nn.Sigmoid())

#     def forward(self, x):
#         x1  = self.input_layer(x) + self.input_skip(x)

#         x2  = self.squeeze_excite1(x1)
#         x2  = self.residual_conv1( x2)

#         x3  = self.squeeze_excite2(x2)
#         x3  = self.residual_conv2( x3)

#         x4  = self.squeeze_excite3(x3)
#         x4  = self.residual_conv3( x4)

#         x5  = self.aspp_bridge(x4)

#         x6  = self.attn1(x3, x5)
#         x6  = self.upsample1(x6)
#         x6  = torch.cat([x6, x3], dim=1)
#         x6  = self.up_residual_conv1(x6)

#         x7  = self.attn2(x2, x6)
#         x7  = self.upsample2(x7)
#         x7  = torch.cat([x7, x2], dim=1)
#         x7  = self.up_residual_conv2(x7)

#         x8  = self.attn3(x1, x7)
#         x8  = self.upsample3(x8)
#         x8  = torch.cat([x8, x1], dim=1)
#         x8  = self.up_residual_conv3(x8)

#         x9  = self.aspp_out(x8)
#         out = self.output_layer(x9)

#         return out
    



# class ResidualConv(nn.Module):
#     def __init__(self, input_dim, output_dim, stride, padding):
#         super(ResidualConv, self).__init__()

#         self.conv_block = nn.Sequential(nn.BatchNorm2d(input_dim),
#                                         nn.ReLU(),
#                                         nn.Conv2d(input_dim,
#                                                   output_dim,
#                                                   kernel_size = 3,
#                                                   stride      = stride,
#                                                   padding     = padding),
#                                         nn.BatchNorm2d(output_dim),
#                                         nn.ReLU(),
#                                         nn.Conv2d(output_dim,
#                                                   output_dim,
#                                                   kernel_size = 3,
#                                                   padding     = 1),
#                                         )
#         self.conv_skip = nn.Sequential(nn.Conv2d(input_dim,
#                                                  output_dim,
#                                                  kernel_size = 3,
#                                                  stride = stride,
#                                                  padding = 1),
#                                        nn.BatchNorm2d(output_dim),
#                                       )

#     def forward(self, x):

#         return self.conv_block(x) + self.conv_skip(x)


# class Upsample(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel, stride):
#         super(Upsample, self).__init__()

#         self.upsample = nn.ConvTranspose2d(input_dim,
#                                            output_dim,
#                                            kernel_size = kernel,
#                                            stride = stride
#                                           )

#     def forward(self, x):
#         return self.upsample(x)


# class Squeeze_Excite_Block(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(Squeeze_Excite_Block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc       = nn.Sequential(nn.Linear(channel, channel // reduction,
#                                                 bias=False),
#                                       nn.ReLU(inplace=True),
#                                       nn.Linear(channel // reduction, channel, bias=False),
#                                       nn.Sigmoid(),
#                                      )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y          = self.avg_pool(x).view(b, c)
#         y          = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


# class ASPP(nn.Module):
#     def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
#         super(ASPP, self).__init__()

#         self.aspp_block1 = nn.Sequential(nn.Conv2d(in_dims,
#                                                    out_dims,
#                                                    3,
#                                                    stride = 1,
#                                                    padding  = rate[0],
#                                                    dilation = rate[0]),
#                                         nn.ReLU(inplace=True),
#                                         nn.BatchNorm2d(out_dims),
#                                        )
#         self.aspp_block2 = nn.Sequential(nn.Conv2d(in_dims,
#                                                    out_dims, 
#                                                    3, 
#                                                    stride   = 1, 
#                                                    padding  = rate[1], 
#                                                    dilation = rate[1]),
#                                          nn.ReLU(inplace=True),
#                                          nn.BatchNorm2d(out_dims),
#                                         )
#         self.aspp_block3 = nn.Sequential(nn.Conv2d(in_dims, 
#                                                    out_dims, 
#                                                    3, 
#                                                    stride   = 1, 
#                                                    padding  = rate[2], 
#                                                    dilation = rate[2]),
#                                          nn.ReLU(inplace = True),
#                                          nn.BatchNorm2d(out_dims),
#                                         )

#         self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
#         self._init_weights()

#     def forward(self, x):
#         x1 = self.aspp_block1(x)
#         x2 = self.aspp_block2(x)
#         x3 = self.aspp_block3(x)
#         out = torch.cat([x1, x2, x3], dim=1)
#         return self.output(out)

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class Upsample_(nn.Module):
#     def __init__(self, scale=2):
#         super(Upsample_, self).__init__()

#         self.upsample = nn.Upsample(mode = "bilinear", 
#                                     scale_factor = scale)

#     def forward(self, x):
#         return self.upsample(x)


# class AttentionBlock(nn.Module):
#     def __init__(self, input_encoder, input_decoder, output_dim):
#         super(AttentionBlock, self).__init__()

#         self.conv_encoder = nn.Sequential(nn.BatchNorm2d(input_encoder),
#                                           nn.ReLU(),
#                                           nn.Conv2d(input_encoder, output_dim, 3, padding=1),
#                                           nn.MaxPool2d(2, 2),
#                                          )

#         self.conv_decoder = nn.Sequential(nn.BatchNorm2d(input_decoder),
#                                           nn.ReLU(),
#                                           nn.Conv2d(input_decoder, output_dim, 3, padding=1),
#                                          )

#         self.conv_attn = nn.Sequential(nn.BatchNorm2d(output_dim),
#                                        nn.ReLU(),
#                                        nn.Conv2d(output_dim, 1, 1),
#                                       )

#     def forward(self, x1, x2):
#         out = self.conv_encoder(x1) + self.conv_decoder(x2)
#         out = self.conv_attn(out)
#         return out * x2



###########################################
class ResUnetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        self.c1 = Stem_Block(   1,   16,  stride = 1)
        self.c2 = ResNet_Block(16,   32,  stride = 2)
        self.c3 = ResNet_Block(32,   64,  stride = 2)
        self.c4 = ResNet_Block(64,  128,  stride = 2)
        self.c5 = ResNet_Block(128, 256,  stride = 2)
        self.c6 = ResNet_Block(256, 512,  stride = 2)
        self.c7 = ResNet_Block(512, 1024, stride = 2)

        self.b1 = ASPP(1024, 2048)

        self.d1 = Decoder_Block([512, 2048], 1024)
        self.d2 = Decoder_Block([256, 1024],  512)
        self.d3 = Decoder_Block([128,  512],  256)
        self.d4 = Decoder_Block([ 64,  256],  128)
        self.d5 = Decoder_Block([ 32,  128],   64)
        self.d6 = Decoder_Block([ 16,   64],   32)

        self.aspp = ASPP(32, 16)

        self.output  = nn.Conv2d(16, 1, kernel_size=1, padding=0)
        self.act_out = nn.Sigmoid()

    def forward(self, inputs):

        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        c6 = self.c6(c5)
        c7 = self.c7(c6)

        b1 = self.b1(c7)

        d1 = self.d1(c7, b1)
        d2 = self.d2(c6, d1)
        d3 = self.d3(c5, d2)
        d4 = self.d4(c4, d3)
        d5 = self.d5(c3, d4)
        d6 = self.d6(c2, d5)

        output = self.aspp(d6)
        output = self.output(output)
        output = self.act_out(output)

        return output
    



class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(nn.Linear(channel, channel // r, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(channel // r, channel, bias=False),
                                 nn.Sigmoid(),
                                )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(),
                                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                )

        self.c2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
                                nn.BatchNorm2d(out_c),
                               )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(nn.BatchNorm2d(in_c),
                                nn.ReLU(),
                                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
                                nn.BatchNorm2d(out_c),
                                nn.ReLU(),
                                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
                               )

        self.c2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
                                nn.BatchNorm2d(out_c),
                               )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y

class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
                                nn.BatchNorm2d(out_c)
                               )

        self.c2 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
                                nn.BatchNorm2d(out_c)
                               )

        self.c3 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
                                nn.BatchNorm2d(out_c)
                               )

        self.c4 = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
                                nn.BatchNorm2d(out_c)
                               )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y

class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(nn.BatchNorm2d(in_c[0]),
                                    nn.ReLU(),
                                    nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
                                    nn.MaxPool2d((2, 2))
                                   )

        self.x_conv = nn.Sequential(nn.BatchNorm2d(in_c[1]),
                                    nn.ReLU(),
                                    nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
                                   )

        self.gc_conv = nn.Sequential(nn.BatchNorm2d(in_c[1]),
                                     nn.ReLU(),
                                     nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                                    )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y

class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d
