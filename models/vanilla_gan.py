import torch


class VanillaGAN:
    def __init__(
        self,
        feature_depth,
        latent_depth,
        generator_hidden_dims=[128, 256, 256],
        discriminator_hidden_dims=[256, 256, 128]
    ):
        self.feature_depth = feature_depth

        self.generator = torch.nn.Sequential()
        self.generator.add_module(
            "linear0",
            torch.nn.Linear(latent_depth, generator_hidden_dims[0])
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
            torch.nn.Linear(generator_hidden_dims[-1], feature_depth)
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
        # self.discriminator.add_module("act%i" % (len(generator_hidden_dims) + 1), torch.nn.Sigmoid())