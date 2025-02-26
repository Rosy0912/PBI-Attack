# 1. Download Vicuna's weights to ./models   (it's a delta version)
# 2. Download LLaMA's weight via: https://huggingface.co/huggyllama/llama-13b/tree/main
# 3. merge them and setup config
# 4. Download the mini-gpt4 compoents' pretrained ckpts
# 5. vision part will be automatically download when launching the model


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import prompt_wrapper
from text_optimize import Text_Attacker

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=3, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")
    parser.add_argument("--n_candidates", type=int, default=400,  # 100
                        help="n_candidates")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

from minigpt_utils import generator

my_generator = generator.Generator(model=model, device=args.gpu_id)

# image initialization
vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')

"""
[Todos] implement text-based attacks (XXXX-4) on this setup.
"""

# ========================================
#                  Attack
# ========================================


import csv

file = open("harmful_corpus/derogatory_corpus.csv", "r")
data = list(csv.reader(file, delimiter=","))
file.close()
targets = []
num = len(data)
for i in range(num):
    targets.append(data[i][0])

# img = Image.open('./output/bad_prompt.bmp').convert('RGB')
img = Image.open('./output/clean.jpeg').convert('RGB')
img = vis_processor(img).unsqueeze(0).to(model.device)

my_attacker = Text_Attacker(args, model, targets, img, device=model.device)

text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_text_attack
offset = prompt_wrapper.minigpt4_chatbot_prompt_offset

n_iters = 500
adv_prompt = my_attacker.attack(text_prompt_template=text_prompt_template, offset=offset,
                                num_iter=n_iters, batch_size=8)
