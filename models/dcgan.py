import numpy as np
import torch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class DCGAN:
    def __init__(
        self,
        feature_shape,
        latent_depth,
        learning_rate=0.0002
    ):
        self.feature_shape = feature_shape
        self.feature_depth = np.prod(feature_shape)
        self.latent_depth = latent_depth
        self.learning_rate = learning_rate

        self.generator = torch.nn.Sequential(
            torch.nn.Linear(self.latent_depth, 256 * 4 * 4, bias=False),
            torch.nn.Unflatten(1, (256, 4, 4)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(
                256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(
                128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2), bias=False
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(
                64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),

            torch.nn.ConvTranspose2d(
                32, 1, kernel_size=(3, 3), padding=(1, 1), bias=False
            ),
            torch.nn.Tanh()
        )

        self.discriminator = torch.nn.Sequential(

            torch.nn.Flatten(),
            torch.nn.Unflatten(1, tuple([1] + self.feature_shape)),

            torch.nn.Conv2d(
                1, 32, kernel_size=(3, 3), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(
                32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            torch.nn.Flatten(),
            torch.nn.Linear(256 * 4 * 4, 1)
        )

        for param in self.generator.parameters():
            torch.nn.init.normal_(param, std=0.02)
        for param in self.discriminator.parameters():
            torch.nn.init.normal_(param, std=0.02)
        
        if torch.cuda.is_available():
            self.generator.to(torch.device("cuda"))
            self.discriminator.to(torch.device("cuda"))

        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
    
    def train_one_step(self, real_samples):
        batch_size = real_samples.shape[0]

        real_samples = FloatTensor(real_samples)
        noises = FloatTensor(np.random.uniform(-1, 1, size=[batch_size, self.latent_depth]))

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
            noises = FloatTensor(
                np.random.uniform(-1, 1, size=[batch_size, self.latent_depth])
            )
        else:
            noises = FloatTensor(noises)

        self.generator.eval()
        faked_samples = self.generator(noises)

        return faked_samples