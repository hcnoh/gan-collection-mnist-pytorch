import os
import json
import pickle
import argparse

import numpy as np
import torch

from utils.mnist_loader import MNISTLoader
from models.gan import GAN
from models.dcgan import DCGAN


def feature_normalize(features):
    return (features - 0.5) / 0.5


def feature_denormalize(features):
    return (features + 1) / 2


def main(model_name):
    if not os.path.isdir(".ckpts"):
        os.mkdir(".ckpts")
    
    if model_name not in ["gan", "dcgan"]:
        print("The model name is wrong!")
        return
    
    ckpt_path = ".ckpts/%s/" % model_name
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[model_name]
    
    loader = MNISTLoader()

    cuda = torch.cuda.is_available()

    if model_name == "gan":
        model = GAN(loader.feature_depth, config["latent_depth"], cuda)
    elif model_name == "dcgan":
        model = DCGAN(loader.feature_shape, config["latent_depth"], cuda)

    steps_per_epoch = (loader.num_train_sets + loader.num_test_sets) // config["batch_size"]

    features = np.vstack([loader.train_features, loader.test_features])
    features = feature_normalize(features)

    generator_losses_epoch = []
    discriminator_losses_epoch = []
    generated_images = []
    for i in range(1, config["num_epochs"] + 1):
        generator_loss_epoch = []
        discriminator_loss_epoch = []
        for _ in range(steps_per_epoch):
            sampled_indices = \
                np.random.choice(
                    loader.num_train_sets + loader.num_test_sets,
                    config["batch_size"],
                    replace=False
                )
            real_samples = features[sampled_indices]

            generator_loss, discriminator_loss = model.train_one_step(real_samples, cuda)

            generator_loss_epoch.append(generator_loss)
            discriminator_loss_epoch.append(discriminator_loss)
        
        generator_loss_epoch = np.mean(generator_loss_epoch)
        discriminator_loss_epoch = np.mean(discriminator_loss_epoch)
        print(
            "Epoch: %i,  Generator Loss: %f,  Discriminator Loss: %f" % \
                (i, generator_loss_epoch, discriminator_loss_epoch)
        )
        generator_losses_epoch.append(generator_loss_epoch)
        discriminator_losses_epoch.append(discriminator_loss_epoch)

        torch.save(
            model.generator.state_dict(), ckpt_path + "generator_%i.ckpt" % i
        )
        torch.save(
            model.discriminator.state_dict(), ckpt_path + "discriminator_%i.ckpt" % i
        )

        faked_samples = feature_denormalize(model.generate(config["batch_size"], cuda))
        generated_images.append(faked_samples)

        with open(ckpt_path + "results.pkl", "wb") as f:
            pickle.dump((generator_losses_epoch, discriminator_losses_epoch, generated_images), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gan",
        help="Type the model name to train. The possible models are [gan, dcgan]"
    )
    args = parser.parse_args()

    main(args.model_name)