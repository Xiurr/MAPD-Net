import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

                                
try:
    from thop import profile
except ImportError:
    profile = None


def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm is much more stable than BatchNorm for 3D medical segmentation with small batch sizes."""
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


                                                                                
                                                   
                                                                                
class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=False):
        super(DeformConv3d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride, stride) if isinstance(stride, int) else stride

        if padding is None:
            self.padding = tuple(k // 2 for k in self.kernel_size)
        else:
            self.padding = (padding, padding, padding) if isinstance(padding, int) else padding

        self.bias = bias
        self.n_points = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]

                                       
        self.offset_conv = nn.Conv3d(
            self.in_channels,
            3 * self.n_points,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias
        )

                        
                                            
                                                     
                                        
                             
        self.regular_conv = nn.Conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,          
            stride=self.stride,
            padding=0,                   
            bias=self.bias
        )

        self.init_offset()

    def init_offset(self):
        self.offset_conv.weight.data.zero_()
        if self.bias:
            self.offset_conv.bias.data.zero_()

    def forward(self, x):
                    
        offset = self.offset_conv(x)
        b, c, d, h, w = x.size()
        N = self.n_points

                   
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.arange(d, device=x.device),
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_d, grid_h, grid_w], dim=-1).float()
        grid = grid.unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(b, 1, 1, 1, 1)

                    
        kd, kh, kw = self.kernel_size
        kernel_offsets_d = torch.arange(-self.padding[0], self.padding[0] + 1, device=x.device)
        kernel_offsets_h = torch.arange(-self.padding[1], self.padding[1] + 1, device=x.device)
        kernel_offsets_w = torch.arange(-self.padding[2], self.padding[2] + 1, device=x.device)

        kd_grid, kh_grid, kw_grid = torch.meshgrid(kernel_offsets_d, kernel_offsets_h, kernel_offsets_w, indexing='ij')
        kernel_offsets = torch.stack([kd_grid.flatten(), kh_grid.flatten(), kw_grid.flatten()], dim=1)
        kernel_offsets = kernel_offsets.permute(1, 0).unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
        kernel_offsets = kernel_offsets.repeat(b, 1, 1, d, h, w)

                   
        offset = offset.view(b, 3, N, d, h, w)
        sampling_grid = grid.unsqueeze(2) + kernel_offsets + offset

                
        sampling_grid_d = sampling_grid[:, 0, ...] / (d - 1) * 2 - 1
        sampling_grid_h = sampling_grid[:, 1, ...] / (h - 1) * 2 - 1
        sampling_grid_w = sampling_grid[:, 2, ...] / (w - 1) * 2 - 1
        normalized_grid = torch.stack([sampling_grid_d, sampling_grid_h, sampling_grid_w], dim=1)

                        
        grid_reshaped = normalized_grid.permute(0, 2, 3, 4, 5, 1).reshape(b, N * d, h, w, 3)
        x_reshaped = x.unsqueeze(2).repeat(1, 1, N, 1, 1, 1).reshape(b, c, N * d, h, w)

        sampled_features = F.grid_sample(
            x_reshaped, grid_reshaped, mode='bilinear', padding_mode='zeros', align_corners=True
        )

                                                 
        sampled_features = sampled_features.view(b, c, N, d, h, w)

                             
        out = 0
        for i in range(N):
            out += self.regular_conv(sampled_features[:, :, i, ...])

        return out


DCNv2_3D = DeformConv3d


                                                                                
                                                
                                                                                
class BoundaryIntensitySynergisticModule(nn.Module):
    def __init__(self, channels):
        super(BoundaryIntensitySynergisticModule, self).__init__()

        self.intensity_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=True)
        self.dcn = DCNv2_3D(channels, channels, kernel_size=3, stride=1, padding=1)
        self.fusion_conv = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, x):
        gate = torch.sigmoid(self.intensity_conv(x))
        dcn_feat = self.dcn(x)
        return self.fusion_conv(torch.cat([gate, dcn_feat], dim=1))


                                                                                
                                            
                                                                                
class MacroContextAnisotropicModule(nn.Module):
    def __init__(self, channels):
        super(MacroContextAnisotropicModule, self).__init__()
                   
        self.aniso_conv_1x1x5 = nn.Conv3d(channels, channels, kernel_size=(1, 1, 5), padding=(0, 0, 2),
                                          bias=False)
        self.aniso_conv_1x5x1 = nn.Conv3d(channels, channels, kernel_size=(1, 5, 1), padding=(0, 2, 0),
                                          bias=False)
        self.aniso_conv_5x1x1 = nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0),
                                          bias=False)
                    
        self.aniso_dilated_conv = nn.Conv3d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False)

        self.fusion_conv = nn.Conv3d(channels * 4, channels, kernel_size=1, bias=False)

    def forward(self, x):
        out1 = self.aniso_conv_1x1x5(x)
        out2 = self.aniso_conv_1x5x1(x)
        out3 = self.aniso_conv_5x1x1(x)
        out4 = self.aniso_dilated_conv(x)

        combined = torch.cat([out1, out2, out3, out4], dim=1)
        return self.fusion_conv(combined)


class IntratumoralHeterogeneityTextureModule(nn.Module):
    def __init__(self, channels):
        super(IntratumoralHeterogeneityTextureModule, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, bias=False)
        self.fusion_conv = nn.Conv3d(channels * 3, channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = torch.cat([self.pool(x), self.conv3(x), self.conv5(x)], dim=1)
        return self.fusion_conv(out)



                                                                                
                                             
                                                                                
def get_3d_laplacian_kernel(device='cpu'):
    center_val = -26.0
    neighbor_val = 1.0
    kernel = torch.full((3, 3, 3), neighbor_val, dtype=torch.float32, device=device)
    kernel[1, 1, 1] = center_val
    return kernel.unsqueeze(0).unsqueeze(0)


class LearnableStructuralGradientModule(nn.Module):
    def __init__(self, channels):
        super(LearnableStructuralGradientModule, self).__init__()
        self.channels = channels

        self.gradient_conv = nn.Conv3d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False
        )
        self._initialize_weights()
        self.avg_pool = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

    def _initialize_weights(self):
        laplacian_kernel = get_3d_laplacian_kernel()
        repeated_kernel = laplacian_kernel.repeat(self.channels, 1, 1, 1, 1)
        self.gradient_conv.weight.data = repeated_kernel
        self.gradient_conv.weight.requires_grad = True

    def forward(self, x):
        gradient = self.gradient_conv(x)
        combined = gradient + x
        pooled = self.avg_pool(combined)
        return pooled * x


class _Stem3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_gn(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_gn(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_gn(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class HeterogeneousPathologyEncoder(nn.Module):
    """HPE: modality-specific private pathology encoder.

    Expected modality order aligned with this repo's dataset channels:
    0: T1, 1: T1ce, 2: T2, 3: FLAIR
    """

    def __init__(self, channels: int, num_modalities: int = 4, use_stem: bool = False):
        super().__init__()
        if num_modalities != 4:
            raise ValueError('HeterogeneousPathologyEncoder currently supports num_modalities=4')

        self.channels = channels
        self.num_modalities = num_modalities

        self.use_stem = bool(use_stem)
        self.stem = _Stem3D(channels) if self.use_stem else nn.Identity()

                                    
        self.t1 = LearnableStructuralGradientModule(channels)
        self.t1ce = BoundaryIntensitySynergisticModule(channels)
        self.t2 = IntratumoralHeterogeneityTextureModule(channels)
        self.flair = MacroContextAnisotropicModule(channels)

    def forward(self, x, modality: int):
        x = self.stem(x)
        if modality == 0:
            return self.t1(x)
        if modality == 1:
            return self.t1ce(x)
        if modality == 2:
            return self.t2(x)
        if modality == 3:
            return self.flair(x)
        raise ValueError(f'Invalid modality index: {modality}')


                                                                                
       
                                                                                
if __name__ == '__main__':
                                                              
    input_shape = (2, 256, 8, 8, 8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device: {device}")
    print(f"Input Shape: {input_shape}")
    print("=" * 70)

          
    dummy_input = torch.randn(input_shape).to(device)

                
    modules_to_test = [
        ("T1ce (Boundary & DCN)", BoundaryIntensitySynergisticModule),
        ("FLAIR (Anisotropic & SE)", MacroContextAnisotropicModule),
        ("T2 (Multi-Scale Texture)", IntratumoralHeterogeneityTextureModule),
        ("T1 (Learnable Gradient)", LearnableStructuralGradientModule)
    ]

    for name, module_cls in modules_to_test:
        print(f"Testing Module: {name}")

             
        model = module_cls(channels=input_shape[1]).to(device)
        model.eval()

                   
        try:
            with torch.no_grad():
                out = model(dummy_input)
            print(f"  Forward Pass: \u2705 Success")
            print(f"  Output Shape: {out.shape}")
            if out.shape != input_shape:
                print(f"  \u26A0 Warning: Shape mismatch!")
        except Exception as e:
            print(f"  Forward Pass: \u274C Failed ({e})")
            continue

                              
        if profile:
            try:
                                                                  
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                print(f"  Params: {params / 1e6:.3f} M")
                print(f"  FLOPs : {flops / 1e9:.3f} G")

                                                                
                                                              
            except Exception as e:
                print(f"  FLOPs calc failed: {e}")
        else:
            print("  FLOPs: thop not installed (pip install thop)")
                                
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Params: {params / 1e6:.3f} M")

        print("-" * 70)

    print("All tests completed.")
