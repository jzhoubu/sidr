from contextlib import nullcontext
from functools import partial
import logging
from typing import List, Union

import torch
from torch import Tensor as T
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertConfig, BatchEncoding, PreTrainedModel

from ..utils.sparse import build_bow_mask, build_topk_mask, elu1p
from ..utils.vis import wordcloud_from_dict
from ..training.ddp_utils import get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



