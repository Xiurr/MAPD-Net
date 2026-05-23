import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, args, nclass=3, ):
        self.args = args
        self.cuda = self.args.cuda
        self.nclass = nclass
        self.VID_branch = None
                                                      
                                                                    
        self.last_details = {}

    def build_loss(self, mode='ce'):
        if mode == 'enumeration':
            print("mode == 'enumeration'")
            return self.EnumerationLoss
        else:
            print(f'Loss {mode} not available.')
            raise NotImplementedError
    
    def _compute_kernel(self, x, y):
                               
        dim = x.size(1)
                             
                                        
                                        
        
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)
        
                             
        euclidean_dist = torch.norm(tiled_x - tiled_y, p=2, dim=2)
        
                                                                                 
        sigma = 1.0 
        kernel = torch.exp(-euclidean_dist / (2 * sigma * sigma))
        return kernel

    def _hsic(self, x, y):
        """
        Hilbert-Schmidt Independence Criterion (HSIC) to measure statistical independence.
        Robust against non-linear dependencies compared to Cosine Similarity.
        """
        Kx = self._compute_kernel(x, x)
        Ky = self._compute_kernel(y, y)
        
        m = x.size(0)             
                            
        H = torch.eye(m, device=x.device) - (1.0 / m) * torch.ones((m, m), device=x.device)
        
                                                
                                                           
        
        hsic = torch.trace(torch.mm(torch.mm(Kx, H), torch.mm(Ky, H))) / ((m - 1) ** 2)
        return hsic

    def EnumerationLoss(self, logits, target, df, df_full, weights=None, extra=None):
                                
        self.last_details = {}

                                                  
                                                                      
        def _dice_loss_with_logits(pred_logits, gt, smooth: float = 1.0):
            pred = torch.sigmoid(pred_logits)
                                       
            dims = tuple(range(2, pred.ndim))
            intersection = torch.sum(pred * gt, dim=dims)
            denom = torch.sum(pred + gt, dim=dims)
            dice = (2.0 * intersection + smooth) / (denom + smooth)
            return 1.0 - dice.mean()

        def _bce_loss_with_logits(pred_logits, gt):
            return F.binary_cross_entropy_with_logits(pred_logits, gt)

                                                
        if extra is not None and isinstance(extra, dict) and 'fused_pred' in extra:
            pred_logits = extra['fused_pred']
        else:
            if isinstance(logits, (list, tuple)):
                pred_logits = torch.mean(torch.stack(list(logits), dim=0), dim=0)
            else:
                pred_logits = logits

        loss_dice = _dice_loss_with_logits(pred_logits, target)
        loss_bce = _bce_loss_with_logits(pred_logits, target)
        loss = loss_dice + loss_bce

                                                 
        try:
            self.last_details['seg/dice'] = loss_dice.detach()
            self.last_details['seg/bce'] = loss_bce.detach()
        except Exception:
            pass

                                                   
        disentangle_on = bool(getattr(self.args.loss, 'disentangle', False))
        if disentangle_on and extra is not None:
            loss = loss + self._midnet_disentanglement_loss(df=df, df_full=df_full, extra=extra)

        return loss

    def _midnet_disentanglement_loss(self, df, df_full, extra):
        """Disentanglement loss in the paper: L_dis = L_cons + L_uni + L_apdm."""
        device = df_full.device
        num_modalities = len(df)
        lambda_dis = float(getattr(self.args.loss, 'lambda_dis', 0.4))
        tau = float(getattr(self.args.loss, 'tau', 0.1))

                                                           
        l_cons = torch.tensor(0.0, device=device)
        if num_modalities > 1:
            for i in range(num_modalities):
                for j in range(i + 1, num_modalities):
                    l_cons = l_cons + torch.mean(torch.abs(df[i] - df[j]))
            l_cons = 2.0 * l_cons / (num_modalities * (num_modalities - 1))

                                                                      
        p_list = extra.get('p_list', None)
        if p_list is None:
            l_uni = torch.tensor(0.0, device=device)
        else:
            gap = [torch.mean(p, dim=(2, 3, 4)) for p in p_list[:num_modalities]]          
            l_uni = torch.tensor(0.0, device=device)
            if len(gap) > 1:
                for i in range(len(gap)):
                    for j in range(i + 1, len(gap)):
                        cos = F.cosine_similarity(gap[i], gap[j], dim=1)
                        l_uni = l_uni + torch.mean(F.relu(cos - tau))
                l_uni = 2.0 * l_uni / (len(gap) * (len(gap) - 1))

                                                      
        pdrm_shared_logits = extra.get('pdrm_shared_logits', None)
        pdrm_weighted_logits = extra.get('pdrm_weighted_logits', None)
        l_apdm = torch.tensor(0.0, device=device)
        cnt = 0
        if pdrm_shared_logits is not None and pdrm_weighted_logits is not None:
            shared_logits = pdrm_shared_logits[0] if isinstance(pdrm_shared_logits, (list, tuple)) else pdrm_shared_logits
            for m in range(min(num_modalities, len(pdrm_weighted_logits))):
                y = torch.full((pdrm_weighted_logits[m].shape[0],), m, dtype=torch.long, device=device)
                l_apdm = l_apdm + F.cross_entropy(shared_logits, y)
                l_apdm = l_apdm + F.cross_entropy(pdrm_weighted_logits[m], y)
                cnt += 1
        if cnt > 0:
            l_apdm = l_apdm / cnt

        l_dis = l_cons + l_uni + l_apdm

        try:
            self.last_details.update({
                'midnet/l_cons': l_cons.detach(),
                'midnet/l_uni': l_uni.detach(),
                'midnet/l_apdm': l_apdm.detach(),
                'midnet/l_dis': l_dis.detach(),
            })
        except Exception:
            pass

        return lambda_dis * l_dis