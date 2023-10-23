from torch import sigmoid
from deepCR.parts import *

class UNet(nn.Module):
    """
    The UNet architecture
    """
    def __init__(self, n_channels, n_classes, hidden=32, 
                num_downs=1, return_type='sigmoid'):
        super(type(self), self).__init__()
        # Initial convolution block
        self.inc = inconv(n_channels, hidden)  
        # List of downsampling blocks
        self.downs = nn.ModuleList([down(hidden * 2**i, hidden * 2**(i+1)) for i in range(num_downs)])
        # List of upsampling blocks
        self.ups = nn.ModuleList([up(hidden * 2**(i+1), hidden * 2**i) for i in range(num_downs)])  
        # Final convolution block
        self.outc = outconv(hidden, n_classes)  

        # what to return
        self.return_type = return_type

    def forward(self, x):
        x1 = self.inc(x)
        downs = [x1]
        del x1
        for down_layer in self.downs:
            x = down_layer(downs[-1])
            downs.append(x)
            del x
        x = downs.pop()
        for up_layer, down_tensor in zip(self.ups, reversed(downs)):
            x = up_layer(x, down_tensor)
            del down_tensor
        x = self.outc(x)

        if self.return_type == 'ori':
            return x
        elif self.return_type == 'sigmoid':
            return sigmoid(x)
        else:
            raise Exception(f'Not supported return_type: {return_type}')