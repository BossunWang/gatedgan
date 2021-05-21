import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_utils import *


# https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        w_variance = torch.sum(torch.pow(x[:, :, :, :-1] - x[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(x[:, :, :-1, :] - x[:, :, 1:, :], 2))
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
        layers += [
            ConvBlock(dim, dim, 3, 1, 1, norm=norm, n_group=n_group, activation=activation, use_affine=use_affine)]
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


class WCT_ZCAPIV2(nn.Module):
    def __init__(self, num_features, mask, groups=1, eps=1e-2, momentum=0.1, w_alpha=0.4, training=True):
        super(WCT_ZCAPIV2, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.groups = groups
        self.alpha = w_alpha
        self.mask = mask
        self.training = training

        self.svdlayer = svdv2.apply

        self.register_buffer('running_content_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_style_mean', torch.zeros(num_features, 1))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_whitening{}".format(i), torch.eye(length, length))
            self.register_buffer("running_coloring{}".format(i), torch.eye(length, length))

    def reset_running_stats(self):
        self.running_content_mean.zero_()
        self.running_style_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def group_feature_transform(self, x, running_mean=None):
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x)) * self.eps
            assert C % G == 0
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            return xg, xxtj, mu
        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = (x - mu)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))

            return xg, mu

    def forward(self, c_A, s_B):
        return self.wct(c_A, s_B)

    def wct(self, c_A, s_B):
        self._check_input_dim(c_A)
        self._check_input_dim(s_B)

        N, C, H, W = c_A.size()
        colored_B_gr_list = []

        if self.training:
            c_A_g, c_A_c_At_j, c_A_mu = self.group_feature_transform(c_A)
            s_B_g, s_B_s_Bt_j, s_B_mu = self.group_feature_transform(s_B)

            for i in range(self.groups):
                c_A_v, c_A_e = self.svdlayer(c_A_c_At_j[i])
                s_B_v, s_B_e = self.svdlayer(s_B_s_Bt_j[i])

                # whitening
                whitening = torch.mm(c_A_v, torch.diag(c_A_e.pow(-0.5)).mm(c_A_v.t()))
                c_A_whitening_gi = torch.mm(whitening, c_A_g[i])

                # coloring
                coloring = torch.mm(s_B_v, torch.diag(s_B_e.pow(0.5)).mm(s_B_v.t()))
                colored_B_coloring_gi = torch.mm(coloring, c_A_whitening_gi)

                colored_B_gr_list.append(colored_B_coloring_gi)

                with torch.no_grad():
                    running_whitening = self.__getattr__('running_whitening' + str(i))
                    running_whitening.data = (1 - self.momentum) * running_whitening.data \
                                             + self.momentum * whitening.data
                    running_coloring = self.__getattr__('running_coloring' + str(i))
                    running_coloring.data = (1 - self.momentum) * running_coloring.data \
                                            + self.momentum * coloring.data

            with torch.no_grad():
                self.running_content_mean = (1 - self.momentum) * self.running_content_mean + self.momentum * c_A_mu
                self.running_style_mean = (1 - self.momentum) * self.running_style_mean + self.momentum * s_B_mu

            colored_B = torch.cat(colored_B_gr_list, dim=0)
            styleize_B = colored_B + s_B_mu.expand_as(colored_B)
            styleize_B = styleize_B.view(C, N, H, W).transpose(0, 1)
        else:
            c_A_g, c_A_mu = self.group_feature_transform(c_A, self.running_content_mean)
            s_B_g, s_B_mu = self.group_feature_transform(s_B, self.running_style_mean)

            for i in range(self.groups):
                # whitening
                whitening = self.__getattr__('running_whitening' + str(i))
                c_A_whitening_gi = torch.mm(whitening, c_A_g[i])

                # coloring
                coloring = self.__getattr__('running_coloring' + str(i))
                colored_B_coloring_gi = torch.mm(coloring, c_A_whitening_gi)

                colored_B_gr_list.append(colored_B_coloring_gi)

            colored_B = torch.cat(colored_B_gr_list, dim=0)
            styleize_B = colored_B + s_B_mu.expand_as(colored_B)
            styleize_B = styleize_B.view(C, N, H, W).transpose(0, 1)

        return styleize_B


class WCT_RobustSVD(nn.Module):
    def __init__(self, n_group, input_dim, w_alpha=0.4, eps=1e-5):
        super(WCT_RobustSVD, self).__init__()
        self.G = n_group
        self.alpha = w_alpha
        self.eps = eps
        self.svdlayer = svdv2.apply

    def forward(self, c_A, s_B):
        return self.wct(c_A, s_B)

    def group_feature_transform(self, x):
        B, C, H, W = x.size()
        n_mem = C // self.G  # 32 if G==8
        xg = x.permute(1, 0, 2, 3).contiguous().view(self.G, n_mem, -1)  # B,C,H,W => C,B,H,W => G,C//G,BHW
        xg_mean = torch.mean(xg, dim=2, keepdim=True)
        xg = xg - xg_mean  # G,C//G,BHW

        cov_xg = torch.bmm(xg, xg.permute(0, 2, 1)).div(B * H * W - 1) \
                 + self.eps * torch.eye(n_mem).unsqueeze(0).to(xg.device)  # G,C//G,C//G

        return xg, cov_xg, xg_mean

    def wct(self, c_A, s_B):
        N, C, H, W = c_A.size()
        c_A_g, cov_c_A, c_A_mean = self.group_feature_transform(c_A)
        s_B_g, cov_s_B, s_B_mean = self.group_feature_transform(s_B)

        # print("c_A_g:", c_A_g.size())
        # print("cov_c_A:", cov_c_A.size())
        # print("s_B_g:", s_B_g.size())
        # print("cov_s_B:", cov_s_B.size())

        colored_B = torch.zeros_like(c_A_g).to(c_A.device)
        for i in range(self.G):
            c_A_v, c_A_e = self.svdlayer(cov_c_A[i])
            s_B_v, s_B_e = self.svdlayer(cov_s_B[i])

            # whitening
            whitening = torch.mm(c_A_v, torch.diag(c_A_e.pow(-0.5)).mm(c_A_v.t()))
            c_A_whitening_gi = torch.mm(whitening, c_A_g[i])

            # coloring
            coloring = torch.mm(s_B_v, torch.diag(s_B_e.pow(0.5)).mm(s_B_v.t()))
            colored_B[i] = torch.mm(coloring, c_A_whitening_gi)

        styleize_B = colored_B + s_B_mean.expand_as(colored_B)
        styleize_B = styleize_B.view(C, N, H, W).transpose(0, 1)
        # print('styleize_B:', styleize_B.size())

        return styleize_B


class WCT(nn.Module):
    def __init__(self, n_group, device, input_dim, mlp_dim, bias_dim, mask, w_alpha=0.4):
        super(WCT, self).__init__()
        self.G = n_group
        self.device = device
        self.alpha = w_alpha
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
    def __init__(self, input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4, device=None, training=True):
        super(Bottleneck, self).__init__()

        curr_dim = input_dim

        # Bottleneck layers
        self.resblocks = nn.ModuleList(
            [ResidualBlock(dim=curr_dim, norm='none', n_group=n_group) for i in range(repeat_num)])
        # self.gdwct_modules = nn.ModuleList(
        #     [WCT(n_group, device, input_dim, mlp_dim, bias_dim, mask) for i in range(repeat_num + 1)])
        self.wct_RobustSVD_modules = nn.ModuleList(
            [WCT_RobustSVD(n_group, input_dim) for i in range(repeat_num + 1)])

    def forward(self, c_A, s_B):
        # Multi-hops
        for i, resblock in enumerate(self.resblocks):
            if i == 0:
                c_A = self.wct_RobustSVD_modules[i](c_A, s_B)

            c_A = resblock(c_A)
            c_A = self.wct_RobustSVD_modules[i + 1](c_A, s_B)

        return c_A


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
            layers += [ConvBlock(curr_dim, curr_dim * 2, 4, 2, 1, norm=norm, activation=activation)]
            curr_dim = curr_dim * 2

        # Bottleneck layers
        for i in range(repeat_num):
            layers += [ResidualBlock(dim=curr_dim, norm=norm, activation=activation)]

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, inputs):
        return [self.main(input) for input in inputs]


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
            layers += [ConvBlock(curr_dim, curr_dim * 2, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)]
            curr_dim = curr_dim * 2

        # Down-sampling layers (keep dim)
        # H/4,W/4,256, H/8,W/8,256, H/16,W/16,256
        for i in range(2):  # original: 2
            layers += [ConvBlock(curr_dim, curr_dim, 4, 2, 1, norm=norm, n_group=n_group, activation=activation)]
        # layers += [nn.AdaptiveAvgPool2d(1)]  # H/16,W/16,256 => 1,1,256

        self.main = nn.Sequential(*layers)
        self.curr_dim = curr_dim

    def forward(self, inputs):
        return [[self.main(input['style']), input['style_label']] for input in inputs]


class Transformer(nn.Module):
    def __init__(self, n_styles
                 , input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num=4
                 , device=None, is_training=True):
        super(Transformer, self).__init__()
        self.transformers = nn.ModuleList(
            [Bottleneck(input_dim, mask, n_group, bias_dim, mlp_dim, repeat_num // 2, device=device, training=is_training)
             for i in range(n_styles)])
        self.n_styles = n_styles
        self.is_training = is_training

    def forward(self, c_A_list, s_B_list):
        if self.is_training:
            # take first feature only when training
            c_A = c_A_list[0]
            s_B = s_B_list[0][0]
            style_label = s_B_list[0][1]

            for (i, v) in enumerate(style_label[0]):
                if v:
                    transformed = self.transformers[i](c_A, s_B)
                    transformed_feature = transformed
        else:
            transformed_feature = torch.zeros_like(c_A_list[0])
            for c_A, s_B_dict in zip(c_A_list, s_B_list):
                s_B = s_B_dict[0]
                style_label = s_B_dict[1]
                for (i, v) in enumerate(style_label[0]):
                    transformed = self.transformers[i](c_A, s_B)
                    transformed_feature += (transformed * v)

        return transformed_feature


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
                 mlp_dim=256, bias_dim=512, content_dim=256, device=None, is_training=True):
        super(Generator, self).__init__()

        self.c_encoder = Content_Encoder(conv_dim, res_blocks_num // 2, norm='in', activation='relu').to(device)
        self.s_encoder = Style_Encoder(conv_dim, n_group, norm='gn', activation='relu').to(device)
        self.transformer = Transformer(n_styles, content_dim, mask, n_group
                                       , bias_dim, mlp_dim, res_blocks_num // 2, device=device,
                                       is_training=is_training).to(device)
        self.decoder = Decoder(content_dim, n_group, norm='ln').to(device)

    def forward(self, content_list, style_dict_list):
        content_features = self.c_encoder(content_list)
        style_list = self.s_encoder(style_dict_list)
        c_A = self.transformer(content_features, style_list)
        output = self.decoder(c_A)
        return output, content_features[0], style_list[0][0]


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
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


if __name__ == '__main__':
    from scipy.linalg import block_diag
    import numpy as np


    def get_block_diagonal_mask(n_member):
        G = n_group
        ones = np.ones((n_member, n_member)).tolist()
        mask = block_diag(ones, ones)
        for i in range(G - 2):
            mask = block_diag(mask, ones)
        return torch.from_numpy(mask).float()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conv_dim = 64
    content_dim = 256
    mlp_dim = 256
    bias_dim = 256
    res_blocks_num = 8
    n_group = 8
    n_styles = 4

    content = torch.rand((1, 3, 256, 256)).to(device)
    style = torch.rand((1, 3, 256, 256)).to(device)
    label = torch.tensor([1, 0, 0, 0]).unsqueeze(0).to(device)
    style_dict = {'style': style, 'style_label': label}
    # The number of blocks in the coloring matrix: G, The number of elements for each block: n_members^2
    n_mem = content_dim // n_group
    # This is used in generators to make the coloring matrix the block diagonal form.
    mask = get_block_diagonal_mask(n_mem).to(device)

    c_encoder = Content_Encoder(conv_dim, res_blocks_num // 2, norm='in', activation='relu').to(device)
    s_encoder = Style_Encoder(conv_dim, n_group, norm='gn', activation='relu').to(device)
    transformer = Transformer(n_styles, content_dim, mask, n_group
                              , bias_dim, mlp_dim, res_blocks_num // 2, device=device).to(device)
    transformer_test = Transformer(n_styles, content_dim, mask, n_group
                                   , bias_dim, mlp_dim, res_blocks_num // 2, is_training=False, device=device).to(
        device)
    decoder = Decoder(content_dim, n_group, norm='ln').to(device)

    content_features = c_encoder([content])
    style_list = s_encoder([style_dict])

    print("content_feature:", content_features[0].size())
    print("style_feature:", style_list[0][0].size())
    print("style_label:", style_list[0][1].size())

    wct_model = WCT_RobustSVD(n_group, content_dim).to(device)

    wct_feature = wct_model(content_features[0], style_list[0][0])
    print('wct_feature:', wct_feature.size())

    c_A = transformer(content_features, style_list)
    print('c_A:', c_A.size())
    #
    # content_features = c_encoder([content, content])
    # label1 = torch.tensor([0.5, 0, 0, 0]).unsqueeze(0).to(device)
    # label2 = torch.tensor([0, 0.5, 0, 0]).unsqueeze(0).to(device)
    # style_dict1 = {'style': style, 'style_label': label1}
    # style_dict2 = {'style': style, 'style_label': label2}
    # style_list = s_encoder([style_dict1, style_dict2])
    # c_A_test = transformer_test(content_features, style_list)
    # print('c_A_test:', c_A_test.size())
    #
    # output = decoder(c_A)
    # print("decoder output:", output.size())
    #
    # generator = Generator(n_styles, conv_dim, res_blocks_num, mask, n_group, mlp_dim, bias_dim, content_dim, device).to(device)
    # output, _, _ = generator([content], [style_dict])
    # print("generator output:", output.size())

    # discriminator = Discriminator().to(device)
    # output_list = discriminator(output)
    # for out in output_list:
    #     print('discriminator out:', out.size())
