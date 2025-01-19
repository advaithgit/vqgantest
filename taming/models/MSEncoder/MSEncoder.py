import torch
import torch.nn as nn
import math

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        if padding == 'same':
            padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            Normalize(out_channels),  # Using GroupNorm instead of BatchNorm to match original
            Swish(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            Normalize(out_channels),
            Swish()
        )
        
    def forward(self, x):
        return self.conv(x)

class BranchBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super(BranchBlock, self).__init__()
        
        # Ensure out_channels is divisible by 32 if it's large enough
        if out_channels >= 32:
            out_channels = ((out_channels + 31) // 32) * 32
            
        #self.block1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        #self.block2 = ConvBlock(out_channels, out_channels, kernel_size, 1)
        self.block1 = ConvBlock(in_channels, out_channels, kernel_size, stride)
        self.block2 = ConvBlock(out_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        return out2 + out1

class MultiStageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_branches=3, stride=1):
        super(MultiStageEncoder, self).__init__()
        
        # Calculate channels per branch, ensuring divisibility by 32 if possible
        min_channels_per_branch = 32 if out_channels >= 96 else max(1, out_channels // num_branches)
        channels_per_branch = ((min_channels_per_branch + 31) // 32) * 32 if min_channels_per_branch >= 32 else min_channels_per_branch
        
        # Create branches
        self.branches = nn.ModuleList([
            BranchBlock(
                kernel_size=3+2*i,
                in_channels=in_channels,
                out_channels=channels_per_branch,
                stride=stride
            ) for i in range(num_branches)
        ])
        
        total_branch_channels = channels_per_branch * num_branches
        
        # Add projection if needed to match desired output channels
        self.proj = None
        if total_branch_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(total_branch_channels, out_channels, 1, 1, 0),
                Normalize(out_channels),
                Swish()
            )
        
        # Add residual connection if input and output channels differ
        self.use_residual = (in_channels == out_channels and stride == 1)
        if not self.use_residual and in_channels != out_channels:
            self.channel_match = nn.Conv2d(in_channels, out_channels, 1, stride=stride)    
    
    def forward(self, x):
        # Process through multi-scale branches
        branch_outputs = [branch(x) for branch in self.branches]
        combined = torch.cat(branch_outputs, dim=1)
        
        # Project to final number of channels
        if self.proj is not None:
            output = self.proj(combined)
        else:
            output = combined
        
        # Add residual if possible
        if self.use_residual:
            output = output + x
        elif hasattr(self, 'channel_match'):
            output = output + self.channel_match(x)
            
        return output


def Normalize(in_channels):
    """
    Adjusted normalization function to handle varying channel counts
    """
    if in_channels >= 32:
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        # For smaller channel counts, use a smaller number of groups
        num_groups = max(1, in_channels // 4)  # Ensure at least 1 group
        return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
