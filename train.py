import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset import WaterDataset
import numpy as np
import os
from generator_model import Generator
from discriminator_model import Discriminator
from tqdm import tqdm
import config
from run_on_patches import split_image_crops
from utils import gradient_penalty, save_checkpoint
from vgg_loss import VGGLoss
from torch.utils.tensorboard import SummaryWriter


def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
):
    loop = tqdm(loader, leave=True)

    for idx, (input_image, target_image) in enumerate(loop):
        high_res = target_image.to(config.DEVICE)
        low_res = input_image.to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_real = disc(low_res, high_res)
            critic_fake = disc(low_res, fake.detach())
            # gp = gradient_penalty(disc, high_res,
            #                       fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                # + config.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(low_res, fake))
            loss_for_vgg = vgg_loss(low_res, fake)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Critic loss", loss_critic.item(),
                          global_step=tb_step)
        tb_step += 1

        # if idx % 100 == 0 and idx > 0:
        #     split_image_crops("/Users/ma107/Desktop/crawl", gen)

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step


def main():
    dataset = WaterDataset(root_dir="C:\\Users\\gearinc\\Downloads\\archive\\wm-nowm")
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=config.NUM_WORKERS,
    )
    gen = Generator(in_channels=3).to(config.DEVICE)
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    writer = SummaryWriter("logs")
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN,
    #         gen,
    #         opt_gen,
    #         config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_DISC,
    #         disc,
    #         opt_disc,
    #         config.LEARNING_RATE,
    #     )

    for epoch in range(config.NUM_EPOCHS):
        tb_step = train_fn(
            loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            g_scaler,
            d_scaler,
            writer,
            tb_step,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    # try_model = True

    # if try_model:
    #     # Will just use pretrained weights and run on images
    #     # in test_images/ and save the ones to SR in saved/
    #     gen = Generator(in_channels=3).to(config.DEVICE)
    #     opt_gen = optim.Adam(
    #         gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN,
    #         gen,
    #         opt_gen,
    #         config.LEARNING_RATE,
    #     )
    #     plot_examples("test_images/", gen)
    # else:
    # This will train from scratch
    main()
