# %%
import glob
import time
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('RUN IN {}'.format(device))


# %%
# FSRCNN
class FSRCNN(nn.Module):
    def __init__(self, channel, nf=4, upscale=4):  ##play attention the upscales
        super(FSRCNN, self).__init__()
        # Feature extractionn
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=nf, kernel_size=5, stride=1,
                               padding=2)  # nf=56.add padding ,make the data alignment
        self.prelu1 = nn.PReLU()

        # Shrinking
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.prelu2 = nn.PReLU()

        # Non-linear Mapping
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu6 = nn.PReLU()

        # Expanding
        self.conv7 = nn.Conv2d(in_channels=12, out_channels=nf, kernel_size=1, stride=1, padding=0)
        self.prelu7 = nn.PReLU()

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=nf, out_channels=channel, kernel_size=9, stride=upscale,
                                            padding=4, output_padding=3)

    def forward(self, x):  #
        out = self.prelu1(self.conv1(x))
        out = self.prelu2(self.conv2(out))
        out = self.prelu3(self.conv3(out))
        out = self.prelu4(self.conv4(out))
        out = self.prelu5(self.conv5(out))
        out = self.prelu6(self.conv6(out))
        out = self.prelu7(self.conv7(out))
        out = self.last_part(out)

        return out




# %%
# 通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# SE注意力机制
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ECA注意力机制
class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# Coordinate Attention注意力机制
class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        w, h = x.shape[3], x.shape[2]
        x_h = nn.AdaptiveAvgPool2d((h, 1))(x).permute(0, 1, 3, 2)
        x_w = nn.AdaptiveAvgPool2d((1, w))(x)

        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out




# %%
# AFSRCNN
class AFSRCNN(nn.Module):
    def __init__(self, channel, nf=16, upscale=4):  ##play attention the upscales
        super(AFSRCNN, self).__init__()
        # Feature extractionn
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=nf, kernel_size=5, stride=1,
                               padding=2)  # nf=56.add padding ,make the data alignment
        self.prelu1 = nn.PReLU()

        # Shrinking
        self.conv2 = nn.Conv2d(in_channels=nf, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.prelu2 = nn.PReLU()

        # Non-linear Mapping
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu6 = nn.PReLU()

        # Expanding
        self.conv7 = nn.Conv2d(in_channels=12, out_channels=nf, kernel_size=1, stride=1, padding=0)
        self.prelu7 = nn.PReLU()

        # Deconvolution
        self.last_part = nn.ConvTranspose2d(in_channels=nf, out_channels=channel, kernel_size=9, stride=upscale,
                                            padding=4, output_padding=3)

        # attention
        self.ca = ChannelAttention(in_planes=12, ratio=3)
        self.sa = SpatialAttention(kernel_size=3)
        self.se = SELayer(channel=12)
        self.eca = ECA_layer(channel=12)
        self.cab1 = CA_Block(nf, reduction=4)
        self.cab2 = CA_Block(12, reduction=4)
        self.cab3 = CA_Block(nf, reduction=4)

    def forward(self, x):
        out = self.prelu1(self.conv1(x))
        out = self.cab1(out)
        out = self.prelu2(self.conv2(out))
        # out = self.prelu3(self.conv3(out))
        # out = self.prelu4(self.conv4(out))
        out = self.cab2(out)
        # out = self.prelu5(self.conv5(out))
        # out = self.prelu6(self.conv6(out))
        out = self.prelu7(self.conv7(out))
        out = self.cab3(out)
        out = self.last_part(out)

        return out





class MyData2(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, index):
        Limg = Image.open(self.paths[index]).convert('RGB')
        Himg = Image.open(self.paths[index].replace('inputs', 'targets')).convert('RGB')

        if self.transform is not None:
            input = self.transform(np.asarray(Limg))
            target = self.transform(np.asarray(Himg))

        return input, target

    def __len__(self):
        return len(self.paths)


train_loader = DataLoader(dataset=MyData2(paths=glob.glob('traindir/inputs/*'),
                                          transform=transforms.Compose([transforms.ToTensor(),
                                                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                                                             (0.5, 0.5, 0.5))])),
                          batch_size=1, shuffle=True, num_workers=0)

test_loader = DataLoader(dataset=MyData2(paths=glob.glob('validdir/inputs/*'),
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5))])),
                         batch_size=1, shuffle=False, num_workers=0)

for batch_x, batch_y in train_loader:
    print(batch_x.shape, batch_y.shape)
    break


# %%
def train(model, train_loader, valid_loader, epochs=20, lr=1e-3, verbose=1):
    start = time.time()
    loss_fun = nn.MSELoss().to(device)
    optimer = optim.Adam(model.parameters(), lr=lr)
    history = {'train loss': [], 'val loss': [], 'best_val': 1e4}
    for epoch in range(1, epochs + 1):
        t1 = time.time()
        epoch_train_loss = []
        epoch_valid_loss = []
        model.train()
        for setp, (input, target) in enumerate(train_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = loss_fun(output, target)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            epoch_train_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            for setp, (input, target) in enumerate(valid_loader):
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss = loss_fun(output, target)
                epoch_valid_loss.append(loss.item())

        epoch_train_loss = np.mean(np.array(epoch_train_loss))
        epoch_valid_loss = np.mean(np.array(epoch_valid_loss))

        if epoch_valid_loss < history['best_val']:
            history['best_val'] = epoch_valid_loss
            torch.save(model.state_dict(), 'models/model_params.pkl')

        t2 = time.time()
        epoch_time = t2 - t1
        if epoch % verbose == 0:
            print(f'Epoch[{epoch}/{epochs}] train loss: %.3f valid loss: %.3f time: %.3f s' % (
            epoch_train_loss, epoch_valid_loss, epoch_time))
    over = time.time()
    print('train total time: %.3f min' % ((over - start) / 60.0))
    return history


model = AFSRCNN(channel=3).to(device)
history = train(model, train_loader, test_loader, epochs=5)





def psnr_score(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def inverse_normalize(img_tensor, rgb_mean=0.5, rgb_std=0.5):
    inv_normalize = transforms.Normalize(
        mean=-rgb_mean / rgb_std,
        std=1 / rgb_std)

    return inv_normalize(img_tensor[0].cpu()).clamp(0, 1).cpu().data.numpy()


def PSNR(model, test_loader):
    psnr = []
    psnr_base = []
    model.eval()
    for input, target in test_loader:
        output = model(input)
        input = inverse_normalize(input).transpose(2, 1, 0)
        output = inverse_normalize(output).transpose(2, 1, 0)
        target = inverse_normalize(target).transpose(2, 1, 0)

        output1 = Image.fromarray(np.uint8(input * 255.)).convert('RGB')
        w, h = output1.size
        output1 = np.asarray(output1.resize(size=(w * 4, h * 4), resample=0)) / 255.0

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.title('output')
        plt.imshow(output)
        plt.subplot(1, 3, 2)
        plt.title('target')
        plt.imshow(target)
        plt.subplot(1, 3, 3)
        plt.title('output1')
        plt.imshow(output1)

        ps1 = psnr_score(output1, target)
        ps2 = psnr_score(output, target)
        psnr_base.append(ps1)
        psnr.append(ps2)

    return np.mean(np.array(psnr_base)), np.mean(np.array(psnr))

     cv2.imshow(relu2)

model = AFSRCNN(channel=3)
model.load_state_dict(torch.load('models/model_params.pkl'))
psnr_base, psnr = PSNR(model, test_loader)
print('PSNR IN TEST: BASE=%.3f dB MODEL=%.3f dB' % (psnr_base, psnr))

# %%



