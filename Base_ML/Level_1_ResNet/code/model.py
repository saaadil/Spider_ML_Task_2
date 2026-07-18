import torch.nn as nn

class resBlock(nn.Module):  # residual block
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)  # conv 1
        self.bn1 = nn.BatchNorm2d(out_c)  # bn 1
        self.relu = nn.ReLU(inplace=True)  # relu 1
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)  # conv 2
        self.bn2 = nn.BatchNorm2d(out_c)  # bn 2
        self.proj = None
        
        if stride != 1 or in_c != out_c:  # if mismatch
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),  # project conv
                nn.BatchNorm2d(out_c)  # project bn
            )

    def forward(self, x):  # math pass
        ident = x  # save skip
        out = self.conv1(x)  # run conv
        out = self.bn1(out)  # run norm
        out = self.relu(out)  # run relu
        
        out = self.conv2(out)  # run conv
        out = self.bn2(out)  # run norm
        
        if self.proj is not None:  # if proj exists
            ident = self.proj(x)  # run skip
            
        out += ident  # add skip
        out = self.relu(out)  # final relu
        
        return out  # return block

class myResNet(nn.Module):  # main resnet
    def __init__(self, s_chans=[64, 128, 256], blks=2, n_cls=10):
        super().__init__()
        assert 2 <= len(s_chans) <= 4  # enforce stages
        assert 1 <= blks <= 3  # enforce blocks
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # first conv
            nn.BatchNorm2d(64),  # stem bn
            nn.ReLU(inplace=True)  # stem relu
        )
        
        self.stages = nn.ModuleList()  # stage list
        in_c = 64  # start channels
        
        for i, out_c in enumerate(s_chans):  # loop stages
            stride = 1 if i == 0 else 2  # calc stride
            stage = self._mk_stage(in_c, out_c, blks, stride)  # build stage
            self.stages.append(stage)  # add stage
            in_c = out_c  # pass forward
            
        self.gap = nn.AdaptiveAvgPool2d(1)  # pool output
        self.fc = nn.Linear(s_chans[-1], n_cls)  # classifier layer
        self._init_w()  # start weights

    def _mk_stage(self, in_c, out_c, n_blks, stride):  # build blocks
        blks = [resBlock(in_c, out_c, stride=stride)]  # first block
        for _ in range(1, n_blks):  # loop rest
            blks.append(resBlock(out_c, out_c, stride=1))  # append blocks
        return nn.Sequential(*blks)  # return stage

    def _init_w(self):  # weight init
        for m in self.modules():  # loop modules
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # he init
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # one norm
                nn.init.constant_(m.bias, 0)  # zero bias
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # small normal
                nn.init.constant_(m.bias, 0)  # zero bias

    def forward(self, x):  # model pass
        x = self.stem(x)  # stem pass
        for s in self.stages:  # loop stages
            x = s(x)  # stage pass
        x = self.gap(x) # pool pass
        x = x.flatten(1) # flatten tensor
        x = self.fc(x)  # linear pass
        return x  # return logits

    def count_p(self):  # count weights
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # sum trainable