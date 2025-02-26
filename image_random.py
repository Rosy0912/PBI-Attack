import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import os
import torch
from tqdm import tqdm
import random
from minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_perspective_score  # , compute_success_rate

model_path = './models/'


def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


class Image_Attacker:
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    def __init__(self, args, model, targets, image, device='cuda', alpha=1 / 255,
                 epsilon=128 / 255, constrained=False, batch_size=8, text_prompt=""):

        self.args = args
        self.model = model
        self.device = device
        self.image = image
        self.alpha = alpha
        self.epsilon = epsilon
        self.constrained = constrained
        self.batch_size = batch_size
        self.text_prompt = text_prompt
        self.prompt = None
        self.generator = generator.Generator(model=self.model, device=self.device)
        self.best_perspective = 1

        self.targets = targets
        self.num_targets = len(targets)

        self.loss_buffer = []

        # freeze and set to eval model:
        self.model.eval()
        self.model.requires_grad_(False)
        self.Break = False

        if not self.constrained:
            self.adv_noise = torch.rand_like(self.image).to(self.device)
            self.adv_noise.requires_grad_(True)
            self.adv_noise.retain_grad()
        else:
            self.adv_noise = torch.rand_like(self.image).to(self.device) * 2 * epsilon - epsilon
            self.x = denormalize(self.image).clone().to(self.device)
            self.adv_noise.data = (self.adv_noise.data + self.x.data).clamp(0, 1) - self.x.data
            self.adv_noise.requires_grad_(True)
            self.adv_noise.retain_grad()

    def train_batch(self, pre_toxicity, text_prompts):

        best_toxicity = -1
        best_adv_noise = torch.zeros_like(self.image).to(self.device)
        response = 'No response'

        K = 5
        for i in range(K):
            perturbation = 2 * torch.rand_like(self.image).to(self.device) - 1
            # perturbation = torch.rand_like(self.image).to(self.device) * 2 * self.epsilon - self.epsilon
            adv_noise = self.adv_noise + perturbation
            adv_noise = self.adv_noise.clamp(0, 1)

            x_adv = normalize(adv_noise)

            self.prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]],
                                                device=self.device)
            self.prompt.img_embs = self.prompt.img_embs * self.batch_size
            self.prompt.update_context_embs()

            response = self.get_response(x_adv)
            toxicity = get_perspective_score(response, return_all=False)

            # toxicity = torch.tensor(-toxicity).to(self.device)

            if toxicity > 1.5:
                print('######### Break!!! - score = %f ##########' % toxicity)
                print('>>> response:', response)
                print('######### === End === ##########')
                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)

            if toxicity > best_toxicity:
                best_toxicity = toxicity
                best_adv_noise = adv_noise

            # res = compute_success_rate(self.text_prompt, response)
            # if res:
            #     self.Break = True

        self.adv_noise = self.adv_noise + best_adv_noise

        return best_toxicity, best_adv_noise, response

    def train(self):

        best_adv_noise = torch.zeros_like(self.image).to(self.device)
        adv_img_prompt = None

        for epoch in tqdm(range(0, self.args.epoch + 1)):

            text_prompts = [self.text_prompt] * self.batch_size

            pre_toxicity = -1 if self.loss_buffer == [] else self.loss_buffer[-1]
            best_toxicity, best_adv_noise, response = self.train_batch(pre_toxicity, text_prompts)

            self.loss_buffer.append(best_toxicity)

            print('\n best_toxicity:', max(best_toxicity, pre_toxicity))

            print("\n toxicity: %f" % best_toxicity)
            print('>>>', response)

            print()

            if epoch % 1 == 0:
                self.plot_loss()

                print('######### Output - Iter = %d ##########' % epoch)

                # x_adv = self.image + self.adv_noise
                x_adv = self.adv_noise + best_adv_noise
                x_adv = normalize(x_adv)
                if epoch % 1 == 0:
                    adv_img_prompt = denormalize(x_adv).detach().cpu()
                    adv_img_prompt = adv_img_prompt.squeeze(0)
                    save_image(adv_img_prompt, '%s/image_random_%d.bmp' % (self.args.save_dir, epoch))

        return best_adv_noise, adv_img_prompt

    def get_response(self, x_adv):
        self.prompt.update_img_prompts([[x_adv]])
        self.prompt.img_embs = self.prompt.img_embs * self.batch_size
        self.prompt.update_context_embs()
        with torch.no_grad():
            response, _ = self.generator.generate(self.prompt)
        if response == '':
            response = 'No response'
        return response

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/img_loss.png' % self.args.save_dir)
        plt.clf()

        torch.save(self.loss_buffer, '%s/loss' % self.args.save_dir)


if __name__ == '__main__':

    import random
    import argparse
    import os
    import numpy as np
    from PIL import Image
    from minigpt_utils import prompt_wrapper
    from minigpt4.common.config import Config
    from minigpt4.common.dist_utils import get_rank
    from minigpt4.common.registry import registry
    import torch.backends.cudnn as cudnn


    def parse_args():
        parser = argparse.ArgumentParser(description="Demo")
        parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
        parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")
        parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
        parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
        parser.add_argument("--constrained", default=False, action='store_true')
        parser.add_argument("--epoch", default=100, action='store_true')

        parser.add_argument("--save_dir", type=str, default='output_random',
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


    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    file = open('./harmful_corpus/derogatory_corpus.csv', "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    targets = []
    num = len(data)
    for i in range(num):
        targets.append(data[i][0])

    # my_attacker = visual_attacker.Attacker(args, model, targets, device=model.device, is_rtp=False)

    # template_img = '../Visual-Adversarial-Examples-Jailbreak-Large-Language-Models-main/adversarial_images/clean.jpeg'
    template_img = './output/bad_prompt_temp_500.bmp'
    img = Image.open(template_img).convert('RGB')
    img = vis_processor(img).unsqueeze(0).to(args.gpu_id)

    text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt_no_text_input

    attacker = Image_Attacker(args, model, targets, img, device=args.gpu_id, batch_size=8,
                              text_prompt=text_prompt_template)
    best_adv_noise, adv_img_prompt = attacker.train()

    save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
    print('[Done]')
