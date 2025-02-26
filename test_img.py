import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from image_optimize import Image_Attacker
import random
import argparse
import os
import numpy as np
from PIL import Image
from minigpt_utils import prompt_wrapper
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

epochs = 1000
batch_size = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--epoch", default=epochs, action='store_true')

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

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


if __name__ == '__main__':
    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========================================
    #             Model Initialization
    # ========================================

    print('>>> Initializing Models')

    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)  # .to(device)

    # Use DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.to(args.gpu_id)
        # model = model.to(device)

    model.eval()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    print('[Initialization Finished]\n')

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    import csv

    # file = open("../Visual-Adversarial-Examples-Jailbreak-Large-Language-Models-main/harmful_corpus/derogatory_corpus.csv", "r")
    file = open('./harmful_corpus/derogatory_corpus.csv', "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    targets = []
    num = len(data)
    for i in range(num):
        targets.append(data[i][0])

    # my_attacker = visual_attacker.Attacker(args, model, targets, device=model.device, is_rtp=False)

    template_img = '../Visual-Adversarial-Examples-Jailbreak-Large-Language-Models-main/adversarial_images/clean.jpeg'
    img = Image.open(template_img).convert('RGB')
    img = vis_processor(img).unsqueeze(0).to(device)

    text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input

    attacker = Image_Attacker(args, model, targets, img, device=device, batch_size=batch_size,
                              text_prompt=text_prompt_template)
    adv_img_prompt = attacker.train()

    save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
    print('[Done]')
