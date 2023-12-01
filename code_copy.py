import torch
import torch.nn as nn

class EncoderNet(nn.Module):
    def __init__(self, layers):
        super(EncoderNet, self).__init__()

        self.dropout = nn.Dropout(0.1)
        
        self.transpose_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.transpose_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.transpose_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.transpose_conv4 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)

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