import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
from modeling.ensemble.unet3d_parallel import UNet as UNetPara
from modeling.HPE import HeterogeneousPathologyEncoder


def _make_gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class PrivateFeatureExtractor(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_down: int = 4, base_channels: Optional[int] = None):
        super().__init__()
        if num_down < 1:
            raise ValueError('num_down must be >= 1')

        if base_channels is None:
                                                                           
            base_channels = max(16, int(out_channels // 16))

        chs: List[int] = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        chs = chs[:num_down]
        if len(chs) < num_down:
            chs = chs + [chs[-1]] * (num_down - len(chs))

        layers: List[nn.Module] = []
        prev = in_channels
        for c in chs:
            layers += [
                nn.Conv3d(prev, c, kernel_size=3, stride=2, padding=1, bias=False),
                _make_gn(c),
                nn.ReLU(inplace=True),
            ]
            prev = c

                                                                                   
        layers += [
            nn.Conv3d(prev, out_channels, kernel_size=1, bias=False),
            _make_gn(out_channels),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PrivateFeatureExtractorParallel(nn.Module):

    def __init__(
        self,
        num_modalities: int,
        out_channels_per_modality: int,
        num_down: int = 4,
        base_channels_per_modality: Optional[int] = None,
    ):
        super().__init__()
        if num_modalities < 1:
            raise ValueError('num_modalities must be >= 1')
        if num_down < 1:
            raise ValueError('num_down must be >= 1')

        self.num_modalities = int(num_modalities)
        self.out_channels_per_modality = int(out_channels_per_modality)

        if base_channels_per_modality is None:
            base_channels_per_modality = max(8, int(out_channels_per_modality // 32))

        chs: List[int] = [
            base_channels_per_modality,
            base_channels_per_modality * 2,
            base_channels_per_modality * 4,
            base_channels_per_modality * 8,
        ]
        chs = chs[:num_down]
        if len(chs) < num_down:
            chs = chs + [chs[-1]] * (num_down - len(chs))

        layers: List[nn.Module] = []

        in_ch = self.num_modalities
        groups = self.num_modalities
        prev_total = in_ch
        for c_per in chs:
            out_total = groups * int(c_per)
            layers += [
                nn.Conv3d(prev_total, out_total, kernel_size=3, stride=2, padding=1, bias=False, groups=groups),
                _make_gn(out_total),
                nn.ReLU(inplace=True),
            ]
            prev_total = out_total

        out_total = groups * self.out_channels_per_modality
        layers += [
            nn.Conv3d(prev_total, out_total, kernel_size=1, bias=False, groups=groups),
            _make_gn(out_total),
            nn.ReLU(inplace=True),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x_all):
                                
        y = self.net(x_all)
        b, c, d, h, w = y.shape
        m = self.num_modalities
        cb = self.out_channels_per_modality
                                                      
        return y.view(b, m, cb, d, h, w)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambd: float = 1.0):
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.lambd)


@dataclass
class MidNetExtra:
    p_in_list: List[torch.Tensor]
    p_list: List[torch.Tensor]
    w_list: List[torch.Tensor]
    pdrm_shared_logits: List[torch.Tensor]
    pdrm_weighted_logits: List[torch.Tensor]

    def to_dict(self):
        return {
            'p_in_list': self.p_in_list,
            'p_list': self.p_list,
            'w_list': self.w_list,
            'pdrm_shared_logits': self.pdrm_shared_logits,
            'pdrm_weighted_logits': self.pdrm_weighted_logits,
        }


class PromptDrivenRegularizationModule(nn.Module):
    def __init__(self, in_channels: int, num_modalities: int = 4, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.num_modalities = num_modalities
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_modalities),
        )

    def forward(self, z_fused):
        z = self.pool(z_fused)
        return self.classifier(z)


class SemanticFingerprint(nn.Module):

    def __init__(self, in_channels: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
                                                                                
            nn.Linear(hidden, 2 * in_channels),
        )

    def forward(self, p_k):
                          
        return self.mlp(self.pool(p_k))


class FingerprintGuidedFusion(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, f_shared, f_private, fingerprint):
        """
        f_shared: [B, C, D, H, W]
        f_private: [B, C, D, H, W]
        fingerprint: [B, 2*C] (scale+bias)
        """
        b, c, d, h, w = f_private.shape
        scale, bias = torch.chunk(fingerprint, 2, dim=1)
        scale = scale.view(b, c, 1, 1, 1)
        bias = bias.view(b, c, 1, 1, 1)

        f_pw = scale * f_private + bias
        cat_feat = torch.cat([f_shared, f_pw], dim=1)
        mask = self.spatial_gate(cat_feat)
        return f_shared + mask * f_pw


class Ensemble(nn.Module):
    def __init__(self, in_channels, out_channels,
                 output='list', exchange=False, feature=False, modality_specific_norm=True, width_ratio=1., sharing=True, **kwargs):
        super().__init__()

               
        self.in_channels = in_channels
        self.output = output
        self.feature = feature
        self.modality_specific_norm = modality_specific_norm
        self.width_ratio = width_ratio
        self.sharing = sharing
        self.module = UNetPara(1, out_channels, num_modalities=in_channels, parallel=True,
                            exchange=exchange, feature=feature, width_multiplier=width_ratio)

        self.debug = bool(kwargs.get('debug', False))
        self._debug_once = True

        self.midnet = bool(getattr(kwargs, 'midnet', False)) if hasattr(kwargs, 'midnet') else bool(kwargs.get('midnet', False))

                                                               
        self._fusion_training_enabled = bool(kwargs.get('fusion_training_enabled', True))

                                                                                                
        bottleneck_channels = int(512 * float(width_ratio)) // 2
        self.private_extractor = None
        self.hpe = HeterogeneousPathologyEncoder(
            channels=bottleneck_channels,
            num_modalities=in_channels,
            use_stem=False,
        ) if self.midnet else None
        self.grl = GradientReversal(lambd=float(kwargs.get('grl_lambda', 1.0))) if self.midnet else None
        pdrm_hidden = int(kwargs.get('pdrm_hidden', 256))
        pdrm_dropout = float(kwargs.get('pdrm_dropout', 0.0))
        self.pdrm = PromptDrivenRegularizationModule(
            in_channels=bottleneck_channels,
            num_modalities=in_channels,
            hidden=pdrm_hidden,
            dropout=pdrm_dropout,
        ) if self.midnet else None
        self.fingerprint = SemanticFingerprint(
            in_channels=bottleneck_channels,
            hidden=pdrm_hidden,
            dropout=pdrm_dropout,
        ) if self.midnet else None
        
        self.fusion_module = FingerprintGuidedFusion(channels=bottleneck_channels) if self.midnet else None
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(bottleneck_channels, max(16, bottleneck_channels // 2), kernel_size=3, padding=1, bias=False),
            _make_gn(max(16, bottleneck_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(max(16, bottleneck_channels // 2), max(16, bottleneck_channels // 4), kernel_size=3, padding=1, bias=False),
            _make_gn(max(16, bottleneck_channels // 4)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(max(16, bottleneck_channels // 4), max(16, bottleneck_channels // 8), kernel_size=3, padding=1, bias=False),
            _make_gn(max(16, bottleneck_channels // 8)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(max(16, bottleneck_channels // 8), max(16, bottleneck_channels // 16), kernel_size=3, padding=1, bias=False),
            _make_gn(max(16, bottleneck_channels // 16)),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(16, bottleneck_channels // 16), out_channels, kernel_size=1),
        ) if self.midnet else None


    def set_fusion_training(self, enabled: bool = True):
        self._fusion_training_enabled = bool(enabled)


    def forward(self, x, channel=[], weights=None):
        x = [x[:, i:i + 1] for i in range(self.in_channels)]
        out, df = self.module(x)

        df_full = torch.mean(torch.stack(df, dim=0), dim=0)

        extra = None
        if self.training and self.midnet:
                                                                           
            p_in_list = [df[i] for i in range(self.in_channels)]
            p_list = [self.hpe(p_in_list[i], modality=i) for i in range(self.in_channels)]
            w_list = [self.fingerprint(p_list[i]) for i in range(self.in_channels)]

                                                                                                  
            shared_logits = self.pdrm(self.grl(df_full))
            pdrm_shared_logits = [shared_logits for _ in range(self.in_channels)]

                                                                       
            pdrm_weighted_logits = []

            for i in range(self.in_channels):
                                     
                w_params = w_list[i]
                scale, bias = torch.chunk(w_params, 2, dim=1)                 

                                                           
                scale = scale.view(scale.shape[0], scale.shape[1], 1, 1, 1)
                bias = bias.view(bias.shape[0], bias.shape[1], 1, 1, 1)

                                                 
                z = df_full.detach() * scale + bias

                          
                logits = self.pdrm(z)         
                pdrm_weighted_logits.append(logits)

            if self.debug and self._debug_once:
                                                                
                assert p_list[0].shape == df[0].shape, f'p_i shape {p_list[0].shape} != df_i shape {df[0].shape}'
                                                                                                                                     
                self._debug_once = False

            extra_obj = MidNetExtra(
                p_in_list=p_in_list,
                p_list=p_list,
                w_list=w_list,
                pdrm_shared_logits=pdrm_shared_logits,
                pdrm_weighted_logits=pdrm_weighted_logits,
            )
            extra = extra_obj.to_dict()

            if self.midnet and self._fusion_training_enabled:
                fused_feat_accum = 0.0
                valid_count = 0

                for i in range(self.in_channels):
                    p_i = p_list[i]
                    w_i = w_list[i]
                    feat_i = self.fusion_module(df_full, p_i, w_i)
                    fused_feat_accum = fused_feat_accum + feat_i
                    valid_count += 1

                if valid_count > 0:
                    fused_feat_mean = fused_feat_accum / valid_count
                    fused_pred = self.decoder(fused_feat_mean)

                                                    
                    target_shape = x[0].shape[2:]
                    if fused_pred.shape[2:] != target_shape:
                        fused_pred = torch.nn.functional.interpolate(
                            fused_pred, size=target_shape, mode='trilinear', align_corners=False
                        )

                    extra['fused_pred'] = fused_pred

        if self.training:
            return out, df, df_full, extra

                                               
        if self.midnet:
                                                                     
                                                                                
                                                                                     
                                                                                          
            
            preserved = list(range(self.in_channels))
            for c in channel:
                preserved.remove(c)

            df_full = torch.mean(torch.stack([df[i] for i in preserved], dim=0), dim=0)
            
                                                                
            fused_feat_accum = 0.0
            valid_count = 0
            
            for i in preserved:
                p_i = self.hpe(df[i], modality=i)                  
                w_i = self.fingerprint(p_i)           
                feat_i = self.fusion_module(df_full, p_i, w_i)
                
                fused_feat_accum = fused_feat_accum + feat_i
                valid_count += 1
            
            if valid_count > 0:
                fused_feat_mean = fused_feat_accum / valid_count
                
                final_pred = self.decoder(fused_feat_mean)

                target_shape = x[0].shape[2:]            
                if final_pred.shape[2:] != target_shape:
                    final_pred = torch.nn.functional.interpolate(
                        final_pred, size=target_shape, mode='trilinear', align_corners=False
                    )

                return final_pred

                                  
        out = torch.stack(out, dim=0)
        preserved = list(range(self.in_channels))
        for c in channel:
            preserved.remove(c)

        return torch.mean(out[preserved], dim=0)



