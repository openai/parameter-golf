import collections,copy,glob,io,lzma,math,os
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
from flash_attn import flash_attn_func as flash_attn_3_func
