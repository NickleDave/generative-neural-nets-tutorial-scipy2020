# generative neural networks

Contributors:
- Celia Cintas
- Pablo Navarro
- David Nicholson

## Short Description

Generative models fit distributions of data, and let us sample from the fit distribution. Recent advances in training neural networks as generative models have produced amazing results, and present many opportunities for researchers working with domain-specific data.

This tutorial is aimed at anyone who wants to leverage those advances in generative neural networks. After introducing generative models and their applications, we'll get you acquainted with neural networks using the framework PyTorch (Keras?!). Then we present two families of models: generative adversarial networks (GANs) and Variational AutoEncoder Networks (VAEs). We'll walk you through training each model type, providing you with tips and tricks we've learned, and we'll demonstrate the pros and cons of each.  And you'll have the chance to test drive cool applications in fields like archeology, neurobiology, computer vision ...

## Long Description
Outline
- Generative Models old school (KDE, PCA, etc.) (30 mins)
  + Why we want Generative models in our research?
  + Basic concepts on prob dist (?) dimension reduction (?)
  + Some cool applications in the domain expert space for generative models. (CT scans augmentation, archeology, ...)
  + Pros/Cons for GANs/VAEs

- Generative Adversarial Networks (90 mins)
  + What are Generative Adversarial Networks? Basic Notations
  + How to Develop Deep Learning Models With Pytorch
    - Types of layers
    - Optimization
    - Loss
    - Loading data
    - Evaluation
    - Visualization
    - Hands-On basic CNN on framework XXX.
  + How to Implement the GAN Training Algorithm (W/ GAN Hacks to Train Stable Models)
    - Generator and Discriminator structures. Type of layers.
    - Loss functions
    - Training. Interaction between both models.
    - Looking at the latent space.
    - Evaluation of GANs.
    - Hands-On Basic 2DGAN (from a to e)
  + GAN Flavours (DCGAN, CGAN, WGAN, 3DGAN, Cycle GAN, ...)
  + Some cool applications in the domain expert space.


- Variational Autoencoder Networks (90 mins)
  + Basic Concepts (KL-diverge, latent vectors, etc.)
  + How to Implement VAEs and training.
    - Encoder/Decoder structures. Type of layers.
    - Loss function and Training.
    - Evaluation & Vis
    - Hands-On Basic VAEs
    - VAEs Flavours
  + Some cool applications in the domain expert space.

## Table of Contents (TOC)

- Introduction
    + [Why generative](notebooks/intro/01-why-generative.ipynb)
    + [Why GANs](notebooks/intro/02-why-GANs.ipynb)
    
- GANs with Pytorch
    + [Discriminator](notebooks/GANs-with-pytorch/01-discriminator.ipynb)
    + [Training SGD](notebooks/GANs-with-pytorch/02-training-SGD.ipynb)
    + [Generator](notebooks/GANs-with-pytorch/03-generator.ipynb)
    + [Training Adversarial](notebooks/GANs-with-pytorch/04-training-adversarial.ipynb)