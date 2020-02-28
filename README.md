# Science in Deep Latent Space:
## Introduction to Generative Adversarial Networks for Researchers

Contributors:
- Celia Cintas
- Pablo Navarro
- David Nicholson

### About
Generative models fit a distribution of data, representing it as a latent space of probability density. Recent advances in training deep neural networks as generative models have resulted in an explosive growth of research and applications. In particular, one newly-developed family of algorithms known as Generative Adversarial Networks (GANs) has achieved impressive breakthroughs on problems previously considered ill-posed. GAN models have become state-of-the-art for many computer vision tasks, such as style transfer, super-resolution, and signal restoration. This tutorial is aimed at anyone who wants to apply GAN models to their work. In this tutorial, we'll start you off with an interactive introduction to generative models and their applications. Then we'll walk you through building and training GANs, providing you with the benefit of our hard-won experience, and the tips and tricks we've learned. Along the way we'll get you acquainted with neural networks using the framework PyTorch. No experience required; we introduce concepts when we need them. Finally, after a crash course in training GANs, we'll expose you to some common use cases. At the same time, we'll give you the chance to test drive cool applications: generating Egyptian hieroglyphics, reconstructing pottery, interpolating birdsong, and much more. After the tutorial, you'll know how to best get started building GAN models of your own data, taking your science to deep latent space.

### Setup
(adapted from https://github.com/deniederhut/Pandas-Tutorial-SciPyConf-2018)
#### 1. Install Python

If you don't already have a working python distribution, you may download one of

* Miniconda ([https://conda.io/miniconda.html](https://conda.io/miniconda.html))
* Python.org  ([https://www.python.org/downloads/](https://www.python.org/downloads/))

You'll need Python 3.

#### 2. Download tutorial materials

This GitHub repository is all that is needed in terms of tutorial content.
The simplest solution is to download the material using this link:

<https://github.com/NickleDave/generative-neural-nets-tutorial-scipy2020>

If you're familiar with Git, you can also clone this repository with:

```sh
git clone git@github.com:NickleDave/generative-neural-nets-tutorial-scipy2020.git
```

It will create a new folder named Pandas-Tutorial-SciPyConf-2018/ with all the
content you will need, including:

- `requirements.txt` - the package requirements for this tutorial
- `check_environment.py` - a script for testing your installation
- `notebooks/` - the Jupyter notebooks we'll use during the tutorial

#### 3. Install required packages

If you are using conda, you can install the necessary packages by opening a terminal and entering the following:

```sh
conda update conda --yes
conda --version  # Should be about 4.5.4
conda env create --file=environment.yml
conda activate gantut
```

If you are using Python from python.org or your system, you can install
the necessary packages by opening a terminal and entering the following:

```sh
# Create a new environment
python3 -m venv gantut
source gantut/bin/activate

pip install -U pip wheel setuptools
pip install -U -r requirements.txt
```

#### 4. Test the installation

To make sure everything was installed correctly, open a terminal,
and change its directory (`cd`) so that your working directory is
`generative-neural-nets-tutorial-scipy2020`. The enter the following:

```sh
python check_environment.py
```

#### 5. Start the notebook

```sh
jupyter notebook
```

## Table of Contents (TOC)

- Introduction
    + [Why generative](notebooks/intro/01-why-generative.ipynb)
    + [Why GANs](notebooks/intro/02-why-GANs.ipynb)
    
- GANs with Pytorch
    + [Discriminator](notebooks/GANs-with-pytorch/01-discriminator.ipynb)
    + [Training SGD](notebooks/GANs-with-pytorch/02-training-SGD.ipynb)
    + [Generator](notebooks/GANs-with-pytorch/03-generator.ipynb)
    + [Training Adversarial](notebooks/GANs-with-pytorch/04-training-adversarial.ipynb)