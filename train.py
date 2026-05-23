"""MAPD-Net core release entry.

This repository is a submission-stage (core) release intended for academic review.
The full training pipeline (configs, dataloaders, trainer, checkpoints) will be
released after paper acceptance.

For now, this script provides a lightweight forward verification that exercises
the core architecture (HPE/APDM/PCFM logic) on random inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from modeling.ensemble.ensemble import Ensemble


def _parse_missing(missing: str) -> List[int]:
    if not missing:
        return []
    out: List[int] = []
    for part in missing.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='MAPD-Net core forward verification')
    parser.add_argument('--patch', type=int, default=80, help='Input patch size D=H=W (paper uses 80).')
    parser.add_argument('--classes', type=int, default=3, help='Number of segmentation classes/channels.')
    parser.add_argument('--width-ratio', type=float, default=0.5, help='Width multiplier for UNet backbone.')
    parser.add_argument(
        '--missing',
        type=str,
        default='',
        help='Comma-separated missing modality indices in {0,1,2,3} (e.g., "1,3").',
    )
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    patch = int(args.patch)
    if patch <= 0:
        raise ValueError('--patch must be > 0')
    if patch < 48:
        raise ValueError('--patch must be >= 48 for the current 4-level UNet downsampling and HPE pooling')

    model = Ensemble(
        in_channels=4,
        out_channels=int(args.classes),
        output='list',
        feature=False,
        width_ratio=float(args.width_ratio),
        midnet=True,
    ).to(device)
    model.eval()

    x = torch.randn(1, 4, patch, patch, patch, device=device)
    missing = _parse_missing(str(args.missing))

    with torch.no_grad():
        y = model(x, channel=missing)

    print(f'device={device}')
    print(f'input={tuple(x.shape)}')
    print(f'missing={missing}')
    print(f'output={tuple(y.shape)}')


if __name__ == '__main__':
    main()
