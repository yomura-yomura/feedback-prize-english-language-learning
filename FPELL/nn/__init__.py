from . import module, datamodule, io, training


def set_seed(seed=3655):
    import numpy as np
    import torch
    import os

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # CuDNNのバックエンド実行のための設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



