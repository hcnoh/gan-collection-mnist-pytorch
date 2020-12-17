# GAN Collection for Generating MNIST Images with PyTorch
This repository is a collection of GANs to generate MNIST dataset. The deep learning framework for this repositoy is PyTorch.

More GAN models will be provided on this repository.

![](/assets/img/README/README_2020-12-05-11-50-38.png)

## Install Dependencies
1. Install Python 3.
2. Install the Python packages in `requirements.txt`. If you use a virtual environment for Python package management, you can install all python packages needed by using the following bash command:

    ```bash
    $ pip install -r requirements.txt
    ```

3. Install PyTorch. The version of PyTorch should be greater or equal than 1.7.0. This repository provides the CUDA usage.

## Training
1. Modify `config.json` as your machine settings.
2. Excute training process by `train.py`. An example of usage for `train.py` are following:

    ```bash
    $ python train.py --model_name=gan
    ```

    The following bash command will help you:

    ```bash
    $ python train.py -h
    ```