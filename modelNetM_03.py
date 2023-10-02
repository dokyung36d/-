import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class plainEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(plainEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=stride)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
    
class plainDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(plainDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)
        
        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2, padding = 2)
            self.bn2 = nn.BatchNorm2d(outChannel) 
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

    
class resEncoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(resEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=stride, padding=stride)
        self.bn1 = nn.BatchNorm2d(outChannel)
        self.conv2 = nn.Conv2d(outChannel, outChannel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannel)
        
        self.downsample = None
        if stride != 1:  
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=stride, padding=1),
                nn.BatchNorm2d(outChannel))
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out
    
class resDecoderBlock(nn.Module):
    def __init__(self, inChannel, outChannel, stride):
    
        super(resDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inChannel)
        
        self.downsample = None
    

        if stride == 1:
            self.conv2 = nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(outChannel)
        else:
            self.conv2 = nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2)
            self.bn2 = nn.BatchNorm2d(outChannel)
            
            self.downsample = nn.Sequential(
                nn.ConvTranspose2d(inChannel, outChannel, kernel_size=1, stride=2, output_padding=1),
                nn.BatchNorm2d(outChannel))   
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out
    
    
class EncoderNet(nn.Module):
    def __init__(self, layers):
        super(EncoderNet, self).__init__()

        self.dropout = nn.Dropout(0.1)
        
        self.transpose_conv1 = nn.ConvTranspose2d(1536, 768, kernel_size=4, stride=2, padding=1)
        self.transpose_conv2 = nn.ConvTranspose2d(768, 384, kernel_size=4, stride=2, padding=1)
        self.transpose_conv3 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1)
        self.transpose_conv4 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1)

        self.hidden_layer1 = nn.Conv2d(768, 768, kernel_size= 6, padding="same")
        self.hidden_layer2 = nn.Conv2d(384, 384, kernel_size= 6, padding="same")
        self.hidden_layer3 = nn.Conv2d(192, 192, kernel_size= 6, padding="same")
        
        self.conv = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        
        self.en_layer1 = self.make_encoder_layer(plainEncoderBlock, 64, 64, layers[0], stride=1)  
        self.en_layer2 = self.make_encoder_layer(resEncoderBlock, 64, 128, layers[1], stride=2)
        self.en_layer3 = self.make_encoder_layer(resEncoderBlock, 128, 256, layers[2], stride=2)
        self.en_layer4 = self.make_encoder_layer(resEncoderBlock, 256, 512, layers[3], stride=2)
        self.en_layer5 = self.make_encoder_layer(resEncoderBlock, 512, 512, layers[4], stride=2)

        self.en_layer_for_image = nn.Conv2d(3, 96, kernel_size= 4 ,stride=1, padding="same")
        # input은 192by 192로 
        
        # weight initializaion with Kaiming method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def make_encoder_layer(self, block, inChannel, outChannel, block_num, stride):
        layers = []
        layers.append(block(inChannel, outChannel, stride=stride))
        for i in range(1, block_num):
            layers.append(block(outChannel, outChannel, stride=1))

        return nn.Sequential(*layers)
    
    def forward(self, input0, input1, input2, input3):
        x = self.transpose_conv1(input3)
        x = self.hidden_layer1(x)
        x = F.leaky_relu(x + input2)
        x = self.dropout(x)

        x = self.transpose_conv2(x)
        x = self.hidden_layer2(x)
        x = F.leaky_relu(x + input1)
        x = self.dropout(x)

        x = self.transpose_conv3(x)
        x = self.hidden_layer3(x)
        x = F.leaky_relu(x + input0) #(1, 192, 96, 96)
        x = self.dropout(x)

        x = F.relu(self.transpose_conv4(x))  #[1, 96, 192, 192]
        x = F.relu(self.bn(self.conv(x)))

        x = F.leaky_relu(self.en_layer1(x))     #128
        x = self.dropout(x)
        x = F.leaky_relu(self.en_layer2(x))     #64
        x = self.dropout(x)
        x = F.leaky_relu(self.en_layer3(x))     #32
        x = self.dropout(x)
        
        x = F.leaky_relu(self.en_layer4(x))     #16
        x = F.leaky_relu(self.en_layer5(x))     #8
        
        return x
    
    
class DecoderNet(nn.Module):
    def __init__(self, layers):
        super(DecoderNet, self).__init__()  

        self.dropout = nn.Dropout(0.1)
        
        self.de_layer5 = self.make_decoder_layer(resDecoderBlock, 512, 512, layers[4], stride=2)
        self.de_layer4 = self.make_decoder_layer(resDecoderBlock, 512, 256, layers[3], stride=2)
        self.de_layer3 = self.make_decoder_layer(resDecoderBlock, 256, 128, layers[2], stride=2)
        self.de_layer2 = self.make_decoder_layer(resDecoderBlock, 128, 64, layers[1], stride=2)
        self.de_layer1 = self.make_decoder_layer(plainDecoderBlock, 64, 64, layers[0], stride=1)
        
        self.conv_end1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=2)
        self.conv_end2 = nn.Conv2d(32, 18, kernel_size=4, stride=1, padding=1)
        self.conv_end3 = nn.Conv2d(18, 13, kernel_size=4, stride=1, padding=1)
        # self.conv_end4 = nn.Conv2d(24, 18, kernel_size=4, stride=1, padding=1)
        # self.conv_end5 = nn.Conv2d(18, 13, kernel_size = 5, stride =1, padding = 1)
                       
        # weight initializaion with Kaiming method
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def make_decoder_layer(self, block, inChannel, outChannel, block_num, stride):

        layers = []
        for i in range(0, block_num-1):
            layers.append(block(inChannel, inChannel, stride=1))
            
        layers.append(block(inChannel, outChannel, stride=stride))
        
        return nn.Sequential(*layers)
                       
    def forward(self, x):
        
        x = self.de_layer5(x)     #8
        x = self.dropout(x)
        x = self.de_layer4(x)     #16
        x = self.dropout(x)
        x = self.de_layer3(x)     #32
        x = self.dropout(x)
        x = self.de_layer2(x)     #64
        x = self.de_layer1(x)     #128   
        
        x = F.leaky_relu(self.conv_end1(x)) #torch.Size([16, 2, 256, 256])
        x = F.leaky_relu(self.conv_end2(x))
        # print("After conv_end2 : ", x.shape)
        x = F.leaky_relu(self.conv_end3(x))
        # print("After conv_end3 : ", x.shape)
        # x = F.leaky_relu(self.conv_end4(x))
        # print("After conv_end4 : ", x.shape)
        # x = self.conv_end5(x)
        # print("After conv_end5 : ", x.shape)
        return x   
        
class ClassNet(nn.Module):
    def __init__(self):
        super(ClassNet, self).__init__()  
        
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.fc = nn.Conv2d(512, 13, 1) # 12개 class + 1 background
                       
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        print("last_shape : ", x.shape)
        x = self.fc(x)
        
        return x  
        
class EPELoss(nn.Module):
    def __init__(self):
        super(EPELoss, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output - target + 1e-16, p=2, dim=1).mean()
        return lossvalue
    
def make_tensor(inputs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = torch.tensor([]).to(device)

    for input in inputs:
        output = torch.cat((output, input), dim = 0)

    return output