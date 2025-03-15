import torch 
import torch.nn as nn
import  torch.nn.functional as F


class UNetEncoder(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(UNetEncoder,self).__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1)
        self.pool = nn.MaxPool3d(2,2)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        skip = x 
        x = self.pool(x)

        return x, skip

class UNetDecoder(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(UNetDecoder,self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=2,stride=2)
        self.conv1 = nn.Conv3d(out_channels * 2,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1)
        

    def forward(self,x,skip):
        x = self.upconv(x)
        
        x = torch.cat([x, skip],dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x 
    
class UNetBottleneck(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(UNetBottleneck,self).__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1)
        self.conv2 = nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        return x 
    

class UnetModel(nn.Module):
    def __init__(self):
        super(UnetModel,self).__init__()

        self.encoder1 = UNetEncoder(3,64)
        self.encoder2 = UNetEncoder(64,128)
        self.encoder3 = UNetEncoder(128,256)
        self.encoder4 = UNetEncoder(256,512)

        self.bottleneck = UNetBottleneck(512,1024)

        self.decoder1 = UNetDecoder(1024,512)
        self.decoder2 = UNetDecoder(512,256)
        self.decoder3 = UNetDecoder(256,128)
        self.decoder4 = UNetDecoder(128,64)

        self.final_conv = nn.Conv3d(64,66,kernel_size=1)
    
    def forward(self,x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x) 

        x = self.bottleneck(x)

        x = self.decoder1(x,skip4)
        x = self.decoder2(x,skip3)
        x = self.decoder3(x,skip2)
        x = self.decoder4(x,skip1)
        
        x = self.final_conv(x)
        x = F.softmax(x, dim=1)

        return x 
