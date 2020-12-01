import numpy as np
import torch


class GAN:
    def __init__(
        self,
        feature_depth,
        latent_depth,
        learning_rate=0.0002,
        generator_hidden_dims=[128, 256, 256],
        discriminator_hidden_dims=[256, 256, 128]
    ):
        self.feature_depth = feature_depth
        self.latent_depth = latent_depth
        self.learning_rate = learning_rate

        self.generator = torch.nn.Sequential()
        self.generator.add_module(
            "linear0",
            torch.nn.Linear(self.latent_depth, generator_hidden_dims[0])
        )
        self.generator.add_module("act0", torch.nn.LeakyReLU())
        for i in range(1, len(generator_hidden_dims)):
            self.generator.add_module(
                "linear%i" % i,
                torch.nn.Linear(generator_hidden_dims[i - 1], generator_hidden_dims[i])
            )
            self.generator.add_module("act%i" % i, torch.nn.LeakyReLU())
        self.generator.add_module(
            "linear%i" % len(generator_hidden_dims),
            torch.nn.Linear(generator_hidden_dims[-1], self.feature_depth)
        )
        self.generator.add_module("act%i" % len(generator_hidden_dims), torch.nn.Tanh())

        self.discriminator = torch.nn.Sequential()
        self.discriminator.add_module(
            "linear0",
            torch.nn.Linear(feature_depth, discriminator_hidden_dims[0])
        )
        self.discriminator.add_module("act0", torch.nn.LeakyReLU())
        for i in range(1, len(discriminator_hidden_dims)):
            self.discriminator.add_module(
                "linear%i" % i, 
                torch.nn.Linear(discriminator_hidden_dims[i - 1], discriminator_hidden_dims[i])
            )
            self.discriminator.add_module("act%i" % i, torch.nn.LeakyReLU())
        self.discriminator.add_module(
            "linear%i" % len(generator_hidden_dims), 
            torch.nn.Linear(discriminator_hidden_dims[-1], 1)
        )

        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
    
    def train_one_step(self, real_samples):
        batch_size = real_samples.shape[0]

        real_samples = torch.tensor(real_samples).float()
        noises = torch.tensor(np.random.normal(size=[batch_size, self.latent_depth])).float()

        self.generator.train()
        self.discriminator.train()

        faked_samples = self.generator(noises)
        faked_scores = self.discriminator(faked_samples)

        generator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=faked_scores, target=torch.ones_like(faked_scores)
        )
        
        self.generator_opt.zero_grad()
        generator_loss.backward()
        self.generator_opt.step()

        real_scores = self.discriminator(real_samples)

        faked_samples = self.generator(noises)
        faked_scores = self.discriminator(faked_samples)

        discriminator_loss = \
            torch.nn.functional.binary_cross_entropy_with_logits(
                input=real_scores, target=torch.ones_like(real_scores)
            ) + \
                torch.nn.functional.binary_cross_entropy_with_logits(
                    input=faked_scores, target=torch.zeros_like(faked_scores)
                )

        self.discriminator_opt.zero_grad()
        discriminator_loss.backward()
        self.discriminator_opt.step()

        return generator_loss.item(), discriminator_loss.item()
    
    def generate(self, batch_size, noises=None):
        if noises == None:
            noises = torch.tensor(
                np.random.normal(size=[batch_size, self.latent_depth])
            ).float()
        else:
            noises = torch.tensor(noises).float()

        self.generator.eval()
        faked_samples = self.generator(noises)

        return faked_samples