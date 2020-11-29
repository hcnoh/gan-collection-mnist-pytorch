import os
import pickle

import numpy as np
import torch

import config

from utils.mnist_loader import MNISTLoader
from models.vanilla_gan import VanillaGAN


def feature_normalize(features):
    return (features - 0.5) / 0.5


def feature_denormalize(features):
    return (features + 1) / 2


def main():
    loader = MNISTLoader()

    model = VanillaGAN(loader.feature_depth, config.LATENT_DEPTH)

    generator_opt = torch.optim.Adam(model.generator.parameters(), lr=0.0002)
    discriminator_opt = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002)

    steps_per_epoch = (loader.num_train_sets + loader.num_test_sets) // config.BATCH_SIZE

    features = np.vstack([loader.train_features, loader.test_features])
    features = feature_normalize(features)

    generator_losses_epoch = []
    discriminator_losses_epoch = []
    generated_images = []
    for i in range(1, config.NUM_EPOCHS + 1):
        generator_loss_epoch = []
        discriminator_loss_epoch = []
        for _ in range(steps_per_epoch):
            sampled_indices = \
                np.random.choice(
                    loader.num_train_sets + loader.num_test_sets,
                    config.BATCH_SIZE,
                    replace=False
                )

            real_samples = torch.tensor(features[sampled_indices]).float()
            noises = torch.tensor(np.random.normal(size=[config.BATCH_SIZE, config.LATENT_DEPTH])).float()

            faked_samples = model.generator(noises)
            faked_scores = model.discriminator(faked_samples)

            generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=faked_scores, target=torch.ones_like(faked_scores)
            )
            
            generator_opt.zero_grad()
            generator_loss.backward()
            generator_opt.step()

            real_scores = model.discriminator(real_samples)

            faked_samples = model.generator(noises)
            faked_scores = model.discriminator(faked_samples)

            discriminator_loss = \
                torch.nn.functional.binary_cross_entropy_with_logits(
                    input=real_scores, target=torch.ones_like(real_scores)
                ) + \
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        input=faked_scores, target=torch.zeros_like(faked_scores)
                    )

            discriminator_opt.zero_grad()
            discriminator_loss.backward()
            discriminator_opt.step()

            generator_loss_epoch.append(generator_loss.item())
            discriminator_loss_epoch.append(discriminator_loss.item())
        
        generator_loss_epoch = np.mean(generator_loss_epoch)
        discriminator_loss_epoch = np.mean(discriminator_loss_epoch)
        print(
            "Epoch: %i,  Generator Loss: %f,  Discriminator Loss: %f" % \
                (i, generator_loss_epoch, discriminator_loss_epoch)
        )
        generator_losses_epoch.append(generator_loss_epoch)
        discriminator_losses_epoch.append(discriminator_loss_epoch)

        torch.save(
            model.generator.state_dict(), config.MODEL_SAVE_PATH + "generator_%i.ckpt" % i
        )
        torch.save(
            model.discriminator.state_dict(), config.MODEL_SAVE_PATH + "discriminator_%i.ckpt" % i
        )

        noises = torch.tensor(np.random.normal(size=[config.BATCH_SIZE, config.LATENT_DEPTH])).float()

        faked_samples = model.generator(noises)
        faked_samples = feature_denormalize(faked_samples)
        generated_images.append(faked_samples)

        with open(config.MODEL_SAVE_PATH + "results.pkl", "wb") as f:
            pickle.dump((generator_losses_epoch, discriminator_losses_epoch, generated_images), f)


if __name__ == "__main__":
    main()