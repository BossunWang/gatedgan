import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight= 1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):    
        w_variance = torch.sum(torch.pow(x[:,:,:,:-1] - x[:,:,:,1:], 2))
        h_variance = torch.sum(torch.pow(x[:,:,:-1,:] - x[:,:,1:,:], 2))
        loss = self.TVLoss_weight * (h_variance + w_variance)
        return loss


# https://github.com/pytorch/pytorch/issues/9160#issuecomment-483048684
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim, norm='in', n_group=32, activation='relu', use_affine=True):
        super(ResidualBlock, self).__init__()
        layers = []
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation=activation, use_affine=use_affine)]
        layers += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation='none', use_affine=use_affine)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.main(x)


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, k, s, p, dilation=False, norm='in', n_group=32,
                        activation='relu', pad_type='mirror', use_affine=True, use_bias=True):
        super(ConvBlock, self).__init__()

        # Init Normalization
        if norm == 'in':
            self.norm = nn.InstanceNorm2d(output_dim, affine=use_affine, track_running_stats=True)
        elif norm == 'ln':
            # LayerNorm(output_dim, affine=use_affine)
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(n_group, output_dim)
        elif norm == 'none':
            self.norm = None

        # Init Activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        # Init pad-type
        if pad_type == 'mirror':
            self.pad = nn.ReflectionPad2d(p)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(p)

        # initialize convolution
        if dilation:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, dilation=p, bias=use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, k, s, bias=use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='ln', n_group=32, activation='relu', use_affine=True):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # Init Normalization
        if norm == 'ln':
            # self.norm = LayerNorm(output_dim, affine=use_affine)
            self.norm = nn.GroupNorm(1, output_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(n_group, output_dim)

        elif norm == 'none':
            self.norm = None

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.01, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU(num_parameters=1, init=0.25)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_block=1, norm='none', n_group=32, activation='relu'):
        super(MLP, self).__init__()
        layers = []
        curr_dim = dim
        layers += [LinearBlock(input_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]

        for _ in range(num_block):
            layers += [LinearBlock(curr_dim, curr_dim, norm=norm, n_group=n_group, activation=activation)]

        layers += [LinearBlock(curr_dim, output_dim, norm='none', activation='none')]  # no output activations
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x.view(x.size(0), -1))


class Get(object):
    def __init__(self, s_CT, C, G, mask):
        self.s_CT = s_CT
        self.C = C
        self.G = G
        self.mask = mask
        self.n_mem = C // G

    def coloring(self):
        X = []  # coloring matrix
        U_arr = []

        for i in range(self.G):  # This is the same with 'i' in Fig.3(b) in the paper.
            # B,n_mem,n_mem
            s_CT_i = self.s_CT[:, (self.n_mem ** 2) * i: (self.n_mem ** 2) * (i + 1)].unsqueeze(2).view(
                self.s_CT.size(0), self.n_mem, self.n_mem)
            D = (torch.sum(s_CT_i ** 2, dim=1,
                           keepdim=True)) ** 0.5  # Compute the comlumn-wise L2 norm of s_CT_i (we assume that D is the eigenvalues) / B,n_mem,n_mem => B,1,n_mem
            U_i = s_CT_i / D  # B,n_mem,n_mem
            UDU_T_i = torch.bmm(s_CT_i, U_i.permute(0, 2, 1))  # B,n_mem,n_mem

            X += [UDU_T_i]
            U_arr += [U_i]

        eigen_s = torch.cat(U_arr, dim=0)  # eigen_s is used in the coloring regularization / B*G,n_mem,n_mem
        X = torch.cat(X, dim=1)  # B,G*n_mem,n_mem
        X = X.repeat(1, 1, self.G)  # B,C,C
        X = self.mask * X

        return X, eigen_s


class WCT(nn.Module):
    def __init__(self, n_group, device, input_dim, mlp_dim, bias_dim, mask, w_alpha=0.4):
        super(WCT, self).__init__()
        self.G = n_group
        self.device = device
        self.alpha = nn.Parameter(torch.ones(1) - w_alpha)
        self.mlp_CT = MLP(input_dim // n_group, (input_dim // n_group) ** 2, dim=mlp_dim, num_block=3, norm='none',
                          n_group=n_group, activation='lrelu')
        self.mlp_mu = MLP(input_dim, bias_dim, dim=input_dim, num_block=1, norm='none', n_group=n_group,
                          activation='lrelu')
        self.mask = mask

    def forward(self, c_A, s_B):
        return self.wct(c_A, s_B)

    def wct(self, c_A, s_B):
        '''
        style_size torch.Size([1, 766])
        mask_size torch.Size([1, 1, 64, 64])
        content_size torch.Size([1, 256, 64, 64])
        W_size torch.size([1,256,256])
        '''

        B, C, H, W = c_A.size()
        n_mem = C // self.G  # 32 if G==8

        s_B_CT = self.mlp_CT(s_B.view(B * self.G, C // self.G, 1, 1)).view(B, -1)  # B*G,C//G,1,1 => B,G*(C//G)**2
        s_B_mu = self.mlp_mu(s_B).unsqueeze(2).unsqueeze(3)

        X_B, eigen_s = Get(s_B_CT, c_A.size(1), self.G, self.mask).coloring()

        eps = 1e-5
        c_A_ = c_A.permute(1, 0, 2, 3).contiguous().view(self.G, n_mem, -1)  # B,C,H,W => C,B,H,W => G,C//G,BHW
        c_A_mean = torch.mean(c_A_, dim=2, keepdim=True)
        c_A_ = c_A_ - c_A_mean  # G,C//G,BHW

        cov_c = torch.bmm(c_A_, c_A_.permute(0, 2, 1)).div(B * H * W - 1) + eps * torch.eye(n_mem).unsqueeze(0).to(
            self.device)  # G,C//G,C//G

        whitend = c_A_.unsqueeze(0).contiguous().view(C, B, -1).permute(1, 0, 2)  # B,C,HW
        colored_B = torch.bmm(X_B, whitend).unsqueeze(3).view(B, C, H, -1)  # B,C,H,W

        return self.alpha * (colored_B + s_B_mu) + (1 - self.alpha) * c_A, cov_c, eigen_s


class Bottleneck(nn.Module):
    def __init__(self, input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4, device=None):
        super(Bottleneck, self).__init__()

        curr_dim = input_dim

        # Bottleneck layers
        self.resblocks = nn.ModuleList(
            [ResidualBlock(dim=curr_dim, norm='none', n_group=n_group) for i in range(repeat_num)])
        self.gdwct_modules = nn.ModuleList(
            [WCT(n_group, device, input_dim, mlp_dim, bias_dim, mask) for i in range(repeat_num + 1)])

    def forward(self, c_A, s_B):
        whitening_reg = []
        coloring_reg = []

        # Multi-hops
        for i, resblock in enumerate(self.resblocks):
            if i == 0:
                c_A, cov, eigen_s = self.gdwct_modules[i](c_A, s_B)
                whitening_reg += [cov]
                coloring_reg += [eigen_s]

            c_A = resblock(c_A)
            c_A, cov, eigen_s = self.gdwct_modules[i + 1](c_A, s_B)
            whitening_reg += [cov]
            coloring_reg += [eigen_s]

        # cov_reg: G,C//G,C//G
        # W_reg: B*G,C//G,C//G
        return c_A, whitening_reg, coloring_reg


class Content_Encoder(nn.Module):
    def __init__(self, conv_dim=64, repeat_num=4, norm='in', activation='relu'):
        super(Content_Encoder, self).__init__()
        layers = []
        # H,W,3 => H,W,64
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm=norm, activation=activation)]

        # Down-sampling layers
        curr_dim = conv_dim
        # H,W,64 => H/2,W/2,128 => H/4,W/4,256
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, activation=activation)]
            curr_dim = curr_dim * 2

        # Bottleneck layers
        for i in range(repeat_num):
            layers += [ResidualBlock(dim=curr_dim, norm=norm, activation=activation)]

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return self.main(x)


class Style_Encoder(nn.Module):
    def __init__(self, conv_dim=64, n_group=32, norm='ln', activation='relu'):
        super(Style_Encoder, self).__init__()
        layers = []
        # H,W,3 => H,W,64
        layers += [ConvBlock(3, conv_dim, 7, 1, 3, norm='none', n_group=n_group, activation=activation)]

        # Down-sampling layers (dim*2)
        curr_dim = conv_dim
        # H,W,64 => H/2,W/2,128 => H/4,W/4,256
        for i in range(2):
            layers += [ConvBlock(curr_dim, curr_dim*2, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)]
            curr_dim = curr_dim * 2

        # Down-sampling layers (keep dim)
        # H/4,W/4,256, H/8,W/8,256, H/16,W/16,256
        for i in range(2):  # original: 2
            layers += [ConvBlock(curr_dim, curr_dim, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)]
        layers += [nn.AdaptiveAvgPool2d(1)] # H/16,W/16,256 => 1,1,256

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, x):
        return [self.main(x['style']), x['style_label']]


class Transformer(nn.Module):
    def __init__(self, n_styles, input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4, device=None, auto_id=True):
        super(Transformer, self).__init__()
        self.transformers = nn.ModuleList(
            [Bottleneck(input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num // 2, device=device)
             for i in range(n_styles)])
        self.n_styles = n_styles
        if auto_id:
            self.transformers.append(Identity())

    def forward(self, c_A, s_B, style_label):
        whitening_reg = None
        coloring_reg = None

        for (i, v) in enumerate(style_label[0]):
            if v and i == self.n_styles:
                transformed = self.transformers[i](c_A)
            elif v:
                transformed, whitening_reg, coloring_reg = self.transformers[i](c_A, s_B)

        # return content transformed by style specific residual block
        return transformed, whitening_reg, coloring_reg


class Decoder(nn.Module):
    def __init__(self, input_dim, n_group, norm='ln'):
        super(Decoder, self).__init__()

        curr_dim = input_dim
        # Up-sampling layers
        layers = []
        for i in range(2):
            layers += [Upsample(scale_factor=2, mode='nearest')]
            layers += [ConvBlock(curr_dim, curr_dim // 2, 5, 1, 2, norm=norm, n_group=n_group)]
            curr_dim = curr_dim // 2

        layers += [ConvBlock(curr_dim, 3, 7, 1, 3, norm='none', activation='tanh')]

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    def __init__(self, n_styles=4, conv_dim=64, res_blocks_num=8, mask=None, n_group=16,
                 mlp_dim=256, bias_dim=512, content_dim=256, device=None):
        super(Generator, self).__init__()

        self.c_encoder = Content_Encoder(conv_dim, res_blocks_num // 2, norm='in', activation='relu').to(device)
        self.s_encoder = Style_Encoder(conv_dim, n_group, norm='gn', activation='relu').to(device)
        self.transformer = Transformer(n_styles, content_dim, mask, n_group
                                       , bias_dim, mlp_dim, res_blocks_num // 2, device=device).to(device)
        self.decoder = Decoder(content_dim, n_group, norm='ln').to(device)

    def forward(self, content, style_dict):
        content_feature = self.c_encoder(content)
        style_feature, style_onehot_label = self.s_encoder(style_dict)
        c_A, whitening_reg, coloring_reg = self.transformer(content_feature, style_feature, style_onehot_label)
        output = self.decoder(c_A)
        return output, whitening_reg, coloring_reg


class Discriminator(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim=3, n_layer=4, gan_type='lsgan', first_dim=64
                 , norm='none', activation='lrelu', num_scales=3, pad_type='mirror'):
        super(Discriminator, self).__init__()
        self.n_layer = n_layer
        self.gan_type = gan_type
        self.dim = first_dim
        self.norm = norm
        self.activation = activation
        self.num_scales = num_scales
        self.pad_type = pad_type
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [ConvBlock(self.input_dim, dim, 4, 2, 1, norm='none'
                            , activation=self.activation, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [ConvBlock(dim, dim * 2, 4, 2, 1, norm=self.norm
                                , activation=self.activation, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2)  # LSGAN
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


if __name__ == '__main__':
    from scipy.linalg import block_diag
    import numpy as np

    def get_block_diagonal_mask(n_member):
        G = n_group
        ones = np.ones((n_member,n_member)).tolist()
        mask = block_diag(ones,ones)
        for i in range(G-2):
            mask = block_diag(mask,ones)
        return torch.from_numpy(mask).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conv_dim = 64
    content_dim = 256
    mlp_dim = 256
    bias_dim = 256
    res_blocks_num = 8
    n_group = 8
    n_styles = 4

    content = torch.rand((1, 3, 128, 128)).to(device)
    style = torch.rand((1, 3, 128, 128)).to(device)
    label = torch.tensor([1, 0, 0, 0, 0]).unsqueeze(0).to(device)
    style_dict = {'style': style, 'style_label': label}
    # The number of blocks in the coloring matrix: G, The number of elements for each block: n_members^2
    n_mem = content_dim // n_group
    # This is used in generators to make the coloring matrix the block diagonal form.
    mask = get_block_diagonal_mask(n_mem).to(device)

    c_encoder = Content_Encoder(conv_dim, res_blocks_num // 2, norm='in', activation='relu').to(device)
    s_encoder = Style_Encoder(conv_dim, n_group, norm='gn', activation='relu').to(device)
    transformer = Transformer(n_styles, content_dim, mask, n_group
                              , bias_dim, mlp_dim, res_blocks_num // 2, device=device).to(device)
    decoder = Decoder(content_dim, n_group, norm='ln').to(device)

    content_feature = c_encoder(content)
    style_feature, style_onehot_label = s_encoder(style_dict)

    print("content_feature:", content_feature.size())
    print("style_feature:", style_feature.size())

    c_A, whitening_reg, coloring_reg = transformer(content_feature, style_feature, style_onehot_label)
    print('c_A:', c_A.size())

    output = decoder(c_A)
    print("decoder output:", output.size())

    generator = Generator(n_styles, conv_dim, res_blocks_num, mask, n_group, mlp_dim, bias_dim, content_dim, device).to(device)
    output, _, _ = generator(content, style_dict)
    print("generator output:", output.size())

    discriminator = Discriminator().to(device)
    output_list = discriminator(output)
    for out in output_list:
        print('discriminator out:', out.size())