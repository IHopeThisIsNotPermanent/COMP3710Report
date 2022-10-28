#modules.py
import torch
import torch.nn as nn

class encoder_component(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        self.layers = [nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 0),
                   nn.BatchNorm2d(out_size),
                   nn.ReLU(),
                   nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 0),
                   nn.BatchNorm2d(out_size),
                   nn.ReLU()]
    
        self.pooling_layer = nn.MaxPool2d((2,2))
        
    def forward(self, inp):
        ret = inp
        for x in self.layers:
            ret = x(ret)
        return ret, self.pooling_layer(ret)
        
class decoder_component(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        
        self.layers1 = [nn.ConvTranspose2d(in_size, out_size, kernel_size = 2, stride = 2, padding = 0),]
        
        self.layers2 = [nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 0),
                        nn.BatchNorm2d(out_size),
                        nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 0),
                        nn.BatchNorm2d(out_size),
                        nn.ReLU()]
    
    def forward(self, inp, pooled):
        ret = inp
        for x in self.layers1:
            ret = x(ret)
        ret = torch.cat([ret, pooled], axis = 1)
        for x in self.layers2:
            ret = x(ret)
        return ret
    
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        
        self.encoder = nn.ModuleList([encoder_component(3,32),
                        encoder_component(32,64),
                        encoder_component(64,128)])
        
        self.base = nn.ModuleList([nn.Conv2d(128, 256, kernel_size = 3, padding = 0),
                     nn.BatchNorm2d(256),
                     nn.ReLU(),
                     nn.Conv2d(256, 256, kernel_size = 3, padding = 0),
                     nn.BatchNorm2d(256),
                     nn.ReLU()])
        
        self.decoder = nn.ModuleList([decoder_component(256,128),
                        decoder_component(128,64),
                         decoder_component(64,32)])
        
        self.final = [nn.Conv2d(32, 1, kernel_size = 1, padding = 0),]
        
    def forward(self, inp):
        pooled = []
        nxt = inp
        for x in self.encoder:
            nxt, pool = x.forward(nxt)
            pooled.append(pool)
        
        for x in self.base:
            nxt = x(nxt)
            
        for x,p in zip(self.decoder, pooled):
            print(nxt.shape, p.shape)
            nxt = x.forward(nxt, p)
            
        for x in self.final:
            nxt = x(nxt)
            
        return nxt
        
        
if __name__ == "__main__":
    inputs = torch.randn((2,3,256,256))
    unt = UNET()
    out = unt.forward(inputs)
    print(out.shape)