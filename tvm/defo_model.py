import torch
from torch import nn
#from transformers import ViTFeatureExtractor, ViTModel


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, stride=1, padding=(1,1), kernels_per_layer=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(1,1), use_dw=False):
        super(ConvBlock, self).__init__()
        if use_dw:
            self.conv = depthwise_separable_conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.act = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = self.act(x)
        #print(x.shape)
        x = self.bn(x)
        #print(x.shape)
        return x

class DefoNet(nn.Module):
    def __init__(self, in_channels, classes, scale_factor=1, use_dw=False):
        super(DefoNet, self).__init__()
        self.block0 = nn.Sequential(
            ConvBlock(in_channels,32//scale_factor,3,stride=1),
            ConvBlock(32//scale_factor,32//scale_factor,3,stride=1),
            nn.MaxPool2d((2,2))
        )
        self.block1 = nn.Sequential(
            ConvBlock(32//scale_factor,64//scale_factor,3,stride=1,use_dw=use_dw),
            ConvBlock(64//scale_factor,64//scale_factor,3,stride=1,use_dw=use_dw),
            ConvBlock(64//scale_factor,64//scale_factor,3,stride=1,use_dw=use_dw),
            nn.MaxPool2d((2,2))
        )
        self.block2 = nn.Sequential(
            ConvBlock(64//scale_factor,128//scale_factor,3, stride=1, use_dw=use_dw),
            ConvBlock(128//scale_factor,128//scale_factor,3,stride=1, use_dw=use_dw),
            ConvBlock(128//scale_factor,128//scale_factor,3,stride=1, use_dw=use_dw),
            nn.MaxPool2d((2,2))
        )
        self.block3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21632//scale_factor,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.35)
        )
        self.fc = nn.Linear(1024,classes)
        self.classifier = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block0(x)
        #print(x.shape)
        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        pred = self.fc(x)
        #print(x.shape)
        pred = self.classifier(pred)
        return pred#[0]

#class ViT_HF(nn.Module):
#    def __init__(self, classes, last_h_shape):
#        super(ViT_HF, self).__init__()
#        self.backbone =  ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
#        self.classifier = nn.Sequential(
#            nn.Flatten(),
#            nn.Linear(last_h_shape[0]*last_h_shape[1],1024),
#            nn.ReLU(inplace=True),
#            nn.BatchNorm1d(1024),
#            nn.Dropout(p=0.35),
#            nn.Linear(1024,classes),
#            nn.Softmax(dim=1)
#        )
#    def forward(self, x):
#        x = self.backbone(x).last_hidden_state
#        #x = self.backbone(**x).last_hidden_state
#        x = self.classifier(x)
#        return x

if __name__ == '__main__':
    from torchstat import stat
    for i in [1]:
        model = DefoNet(3, 2, scale_factor=i, use_dw=False)
        stat(model, (3,108,108))
        dummy = torch.randn(100,3,108,108)
        print(model(dummy).shape)