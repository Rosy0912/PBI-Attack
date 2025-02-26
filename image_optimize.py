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
from utils import get_perspective_score, compute_success_rate

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

        self.loss_buffer_target = []
        self.loss_buffer_perspective = []

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

    def train_batch(self, targets, text_prompts):

        # optimize D

        # optimize G
        for i in range(5):
            x_adv = normalize(self.adv_noise)

            self.prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            self.prompt.img_embs = self.prompt.img_embs * self.batch_size
            self.prompt.update_context_embs()

            target_loss = self.attack_loss(self.prompt, targets)
            target_loss.backward()

            self.adv_noise.data = (self.adv_noise.data - self.alpha * self.adv_noise.grad.detach().sign()).clamp(0, 1)
            self.adv_noise.grad.zero_()
            self.model.zero_grad()

        # image loss
        for i in range(1):
            x_adv = normalize(self.adv_noise)

            self.prompt = prompt_wrapper.Prompt(model=self.model, text_prompts=text_prompts, img_prompts=[[x_adv]])
            self.prompt.img_embs = self.prompt.img_embs * self.batch_size
            self.prompt.update_context_embs()

            response = self.get_response(x_adv)
            res_lis = get_perspective_score(response, return_all=True)

            loss_jail = torch.tensor(-sum(res_lis), requires_grad=True).to(self.device)
            loss_jail.backward()

            final_loss = torch.tensor(target_loss.item() + loss_jail.item(), requires_grad=True).to(self.device)
            final_loss.backward()

            self.adv_noise.data = (self.adv_noise.data - self.alpha * self.adv_noise.grad.detach().sign()).clamp(0, 1)
            self.adv_noise.grad.zero_()
            self.model.zero_grad()

            if -loss_jail.item() > 1.5:
                print('######### Break!!! - score = %f ##########' % -loss_jail.item())
                print(
                    '>>> TOXICITY: %f, SEVERE_TOXICITY: %f, IDENTITY_ATTACK: %f, INSULT: %f, PROFANITY: %f, THREAT: %f, SEXUALLY_EXPLICIT: %f, FLIRTATION: %f' %
                    res_lis)
                print('>>> response:', response)
                print('######### === End === ##########')
                adv_img_prompt = denormalize(x_adv).detach().cpu()
                adv_img_prompt = adv_img_prompt.squeeze(0)

                if -loss_jail.item() > self.best_perspective:
                    self.best_perspective = -loss_jail.item()
                    save_image(adv_img_prompt, '%s/Best_Break.bmp' % self.args.save_dir)

                if compute_success_rate(self.text_prompt, response):
                    self.Break = True
        return target_loss, loss_jail, final_loss, response, res_lis

    def train(self):

        for epoch in tqdm(range(0, self.args.epoch + 1)):

            if self.Break:
                break

            batch_targets = random.sample(self.targets, self.batch_size)
            text_prompts = [self.text_prompt] * self.batch_size

            target_loss, loss_jail, final_loss, response, res_lis = self.train_batch(
                batch_targets, text_prompts)

            self.loss_buffer_target.append(target_loss.item())
            self.loss_buffer_perspective.append(res_lis)

            print("\n target_loss: %f, perspective: %f, final_loss: %f" % (
                target_loss.item(), loss_jail.item(), final_loss.item()))
            print(
                '>>> TOXICITY: %f, SEVERE_TOXICITY: %f, IDENTITY_ATTACK: %f, INSULT: %f, PROFANITY: %f, THREAT: %f, SEXUALLY_EXPLICIT: %f, FLIRTATION: %f' %
                res_lis)

            if epoch % 5 == 0:
                self.plot_loss()

                print('######### Output - Iter = %d ##########' % epoch)

                # x_adv = self.image + self.adv_noise
                x_adv = self.adv_noise
                x_adv = normalize(x_adv)

                print('>>>', response)
                if epoch % 20 == 0:
                    adv_img_prompt = denormalize(x_adv).detach().cpu()
                    adv_img_prompt = adv_img_prompt.squeeze(0)
                    save_image(adv_img_prompt, '%s/bad_prompt_temp_%d.bmp' % (self.args.save_dir, epoch))

        print('\n\n Final response\n >>>', response)
        return adv_img_prompt, response

    def get_response(self, x_adv):
        self.prompt.update_img_prompts([[x_adv]])
        self.prompt.img_embs = self.prompt.img_embs * self.batch_size
        self.prompt.update_context_embs()
        with torch.no_grad():
            response, _ = self.generator.generate(self.prompt)
        if response == '':
            response = 'No response'
        return response

    def attack_loss(self, prompts, targets):

        context_embs = prompts.context_embs

        if len(context_embs) == 1:
            context_embs = context_embs * len(targets)  # expand to fit the batch_size

        assert len(context_embs) == len(
            targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

        batch_size = len(targets)
        self.model.llama_tokenizer.padding_side = "right"  # module

        to_regress_tokens = self.model.llama_tokenizer(  # module
            targets,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,  # module
            add_special_tokens=False
        ).to(self.device)
        to_regress_embs = self.model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)  # module

        bos = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.bos_token_id
        bos_embs = self.model.llama_model.model.embed_tokens(bos)  # module

        pad = torch.ones([1, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.model.llama_tokenizer.pad_token_id
        pad_embs = self.model.llama_model.model.embed_tokens(pad)  # module

        T = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.model.llama_tokenizer.pad_token_id, -100
        )  # module

        pos_padding = torch.argmin(T, dim=1)  # a simple trick to find the start position of padding

        input_embs = []
        targets_mask = []

        target_tokens_length = []
        context_tokens_length = []
        seq_tokens_length = []

        for i in range(batch_size):

            pos = int(pos_padding[i])
            if T[i][pos] == -100:
                target_length = pos
            else:
                target_length = T.shape[1]

            targets_mask.append(T[i:i + 1, :target_length])
            input_embs.append(to_regress_embs[i:i + 1, :target_length])  # omit the padding tokens

            context_length = context_embs[i].shape[1]
            seq_length = target_length + context_length

            target_tokens_length.append(target_length)
            context_tokens_length.append(context_length)
            seq_tokens_length.append(seq_length)

        max_length = max(seq_tokens_length)

        attention_mask = []

        for i in range(batch_size):
            # masked out the context from loss computation
            context_mask = (
                torch.ones([1, context_tokens_length[i] + 1],
                           dtype=torch.long).to(self.device).fill_(-100)  # plus one for bos
            )

            # padding to align the length
            num_to_pad = max_length - seq_tokens_length[i]
            padding_mask = (
                torch.ones([1, num_to_pad],
                           dtype=torch.long).to(self.device).fill_(-100)
            )

            targets_mask[i] = torch.cat([context_mask, targets_mask[i], padding_mask], dim=1)
            input_embs[i] = torch.cat([bos_embs, context_embs[i], input_embs[i],
                                       pad_embs.repeat(1, num_to_pad, 1)], dim=1)
            attention_mask.append(torch.LongTensor([[1] * (1 + seq_tokens_length[i]) + [0] * num_to_pad]))

        targets = torch.cat(targets_mask, dim=0).to(self.device)
        inputs_embs = torch.cat(input_embs, dim=0).to(self.device)
        attention_mask = torch.cat(attention_mask, dim=0).to(self.device)

        outputs = self.model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            output_attentions=True,
        )
        loss = outputs.loss

        return loss

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer_target)

        x_ticks = list(range(0, num_iters))

        # Plot and label the training and validation loss values
        plt.plot(x_ticks, self.loss_buffer_target, label='Target Loss')

        # Add in a title and axes labels
        plt.title('Loss Plot')
        plt.xlabel('Iters')
        plt.ylabel('Loss')

        # Display the plot
        plt.legend(loc='best')
        plt.savefig('%s/img_loss.png' % self.args.save_dir)
        plt.clf()

        torch.save(self.loss_buffer_target, '%s/loss' % self.args.save_dir)
