import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DCFTLinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, kernel_size=4, stride=2):
        """
        Wraps a pre-trained nn.Linear layer with a DCFT (Deconvolution in Subspace) mechanism.
        """
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # 1. Freeze original weights
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
            
        # 2. DCFT Subspace Configuration
        self.subspace_dim = self.in_features // stride
        
        # A: Down-projection
        self.down_proj = nn.Linear(self.in_features, self.subspace_dim, bias=False)
        
        # d: 1D Deconvolution layer
        self.deconv = nn.ConvTranspose1d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=kernel_size // 2 
        )
        
        # B: Up-projection
        deconv_out_len = (self.subspace_dim - 1) * stride - 2 * (kernel_size // 2) + kernel_size
        self.up_proj = nn.Linear(deconv_out_len, self.out_features, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight) # Zero init ensures initial output perfectly matches base model
        nn.init.normal_(self.deconv.weight, std=0.02)

    def forward(self, x):
        # Base forward pass (Frozen)
        base_out = F.linear(x, self.weight, self.bias)
        
        # Reshape for DCFT
        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features) 
        
        # 1. Project to subspace
        subspace_x = self.down_proj(x_flat) 
        
        # 2. Deconvolution (requires Batch, Channels, Length shape)
        subspace_x = subspace_x.unsqueeze(1)
        deconv_x = self.deconv(subspace_x).squeeze(1) 
        
        # 3. Project up and match original tensor dimensions
        dcft_out = self.up_proj(deconv_x)
        dcft_out = dcft_out.view(*orig_shape[:-1], self.out_features)
        
        return base_out + dcft_out


def inject_dcft_into_unet(module, target_modules=["to_q", "to_k", "to_v", "to_out.0"], kernel_size=4, stride=2):
    """
    Recursively replaces target nn.Linear layers in the UNet with DCFTLinear.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
            dcft_layer = DCFTLinear(child, kernel_size=kernel_size, stride=stride)
            setattr(module, name, dcft_layer)
        else:
            inject_dcft_into_unet(child, target_modules, kernel_size, stride)