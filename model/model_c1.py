import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .GDN import GDN

class ImgComNet(nn.Module):
    def __init__(self, N, M, model, inplace=False):
        super(ImgComNet, self).__init__()
        self.N = N
        self.M = M
        self.resnet = nn.Sequential(*list(model.children())[:-2])

        self.g_analysis = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2)),
            ('GDN1', GDN(N, inverse=False)),
            ('conv2', nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2)),
            ('GDN2', GDN(N, inverse=False)),
            ('conv3', nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2)),
            ('GDN3', GDN(N, inverse=False)),
            ('conv4', nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2))
        ]))

        self.g_synthesis = nn.Sequential(OrderedDict([
            ('Unconv1', nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('IGDN1', GDN(N, inverse=True)),
            ('Unconv2', nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('IGDN2', GDN(N, inverse=True)),
            ('Unconv3', nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1)),
            ('IGDN3', GDN(N, inverse=True)),
            ('Unconv4', nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2, padding=2, output_padding=1))
        ]))

        self.prior = nn.Sequential(OrderedDict([
            ('Sps4', nn.Softplus()),
            ('conv5b', nn.Conv2d(M, 1, kernel_size=1, stride=1, padding=0)),
            ('Sps5', nn.Softplus()),
            ('upsp5', nn.Upsample(scale_factor=16, mode='bilinear'))
        ]))

        self.percept_gen = nn.Sequential(OrderedDict([
            ('conv6', nn.Conv2d(10, 3, kernel_size=1, stride=1, padding=0)),
            ('GDN6', GDN(3, inverse=False))
        ]))

        self.fusion = nn.Sequential(OrderedDict([
            ('maxp9', nn.MaxPool2d(8))
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('fc13', nn.Linear(512, 1))
        ]))

    def para_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def loss_build(self, x_hat, x):
        distortion = F.mse_loss(x_hat, x, size_average=True)
        return distortion

    def forward_once(self, x):
        y = self.g_analysis(x)
        x_hat = self.g_synthesis(y)
        err = x - x_hat
        res = err * err
        res = torch.mean(res, dim=1, keepdim=True)
        prior = self.prior(y)
        prior = prior * prior
        return res, prior, x_hat

    def forward(self, x_left, x_right, label, requires_loss):
        left_res, left_prior, left_rcon = self.forward_once(x_left)
        right_res, right_prior, right_rcon = self.forward_once(x_right)

        left_likelihood = (right_res + 5e-21) / (left_res + right_res + 1e-20)
        right_likelihood = (left_res + 5e-21) / (left_res + right_res + 1e-20)
        left_prior_pro = (left_prior + 5e-21) / (left_prior + right_prior + 1e-20)
        right_prior_pro = (right_prior + 5e-21) / (left_prior + right_prior + 1e-20)

        percept_in = torch.cat((left_prior_pro, left_likelihood, x_left, right_prior_pro, right_likelihood, x_right), dim=1)
        percept_out = self.percept_gen(percept_in)

        res_out = self.resnet(percept_out)
        fc_in = self.fusion(res_out)
        fc_in = fc_in.view(fc_in.size()[0], -1)
        score = self.fc(fc_in)


        if requires_loss:
            return score, label, self.loss_build(score, label)
        else:
            return score, label


