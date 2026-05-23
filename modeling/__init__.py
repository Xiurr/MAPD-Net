                                      
from modeling.ensemble.ensemble import Ensemble

def build_model(args, nclass, nchannels, model='unet', recons=False):
    if model == 'ensemble':
        return Ensemble(
            nchannels,
            nclass,
            output=args.output,
            exchange=args.exchange,
            feature=args.feature,
            width_ratio=args.width_ratio,
            modality_specific_norm=args.modality_specific_norm,
            sharing=args.sharing,
            midnet=getattr(args, 'midnet', False),
            pdrm_hidden=getattr(args, 'pdrm_hidden', 256),
            pdrm_dropout=getattr(args, 'pdrm_dropout', 0.0),
        )
    else:
        raise NotImplementedError