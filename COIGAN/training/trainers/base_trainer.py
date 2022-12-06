import copy
import logging
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler


from COIGAN.utils.common_utils import add_prefix_to_keys, average_dicts, set_requires_grad, flatten_dict
from COIGAN.utils.ddp_utils import get_has_ddp_rank