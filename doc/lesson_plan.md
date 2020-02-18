## Lesson plan

+ **why GANs**, 30m
  - *instruction*: define generative model
    + For the purposes of this tutorial, "any model that takes a training set, consisting of samples drawn from a distribution pdata, and learns to represent an estimate of that distribution somehow." [Goodfellow 2016]
    + demo example: HMM model fit with toy data
  - *instruction*: why do we want generative models in our research?
    + 1. test "our ability to represent and manipulate high-dimensional probability distributions" [Goodfellow 2016]
    + 2. "many tasks intrinsically require realistic generation of samples from some distribution." [Goodfellow 2016]
      - demo: GalaxyGAN (https://arxiv.org/pdf/1702.00403.pdf) or similar
      - *hands-on* example: single image super-resolution
  - *instruction*, 10m: what problems do GANs solve
    + present taxonomy of generative models: explicit v implicit
    + models that are intractable / computationally expensive to fit
      -e.g. require variational and/or Monte Carlo approximation

*break*, 5 m

+ **what are GANs, and what actually is a neural network?** 45m
  - *instruction*, 15m: what is a GAN
    + *hands-on* example: load trained generator network, producing samples; explain: this is our goal!
    + instruction, 10 m: intuition behind generative adversarial training
      - adversarial game between generator and discriminator
        + role of the generator: generate samples that fool discriminator
        + input is vectors from latent space
        + role of discriminator: classify samples as real (data) or fake (output from generator)
      - build intuition with figures, concept maps
  - *instruction*, 30m: crash course in neural networks
    + along with live demo in Numpy
        - an artificial neuron with inputs, weights, bias, activation and output
        - a fully connected layer of artificial neurons
        - a multi-layer perceptron with input, hidden, and output layers
        - simplest form of fitting with Stochastic Gradient Descent
    + *hands-on* exercises, 5m: writing and training a perceptron with Numpy

*break*, 10m. Total so far, 1h30m

+ **building and training your first GAN with PyTorch**, 1h30m
  - *instruction*, 40m: the Discriminator as "just" a standard neural network
    + the job of the Discriminator is to classify "real" and "fake" ~ supervised learning
    + live-coding demo, 15m: building networks by sub-classing torch.nn.module
      - demonstration, 10 m: building a neural network with PyTorch, aka hello nn.module world
      - demonstration: building a network by sub-classing nn.Module
      - defining the __init__ method
      - declaring layers
        + self.fc1 = linear()
      - demonstration: where are the weights? self.layer.parameters()
      - defining the forward method
        + passing input to layers, returning output
      - exercise, 5 m: make a network that combines an input $x$ with another input $z$ through separate fully-connected layers    
        + explain that this was used for first conditional GANs
      - soft, to the max: getting p(real)

    + live-coding demo, 20m: writing a supervised learning training loop
      - how do we fit this thing?
      - how do we train a network, period?
      - train with MNIST (or some simple dataset) using SGD
      - introduce datasets n' dataloaders: your friends
      - creating a criterion
      - in the bad old days, we made batches by hand from artisanal numpy matrices
      - writing the training loop
      - computing the loss
      - logging the loss

    + *hands-on* exercises: adding to your Discriminator, putting some label smoothing in your training loop (GAN hack!)

  - *instruction*, 40m: the Generator
    + live-coding demo, 15m: modifying our Discriminator to build a Generator
      - second verse, same at the first  -- making a Generator class
      - exercise: try to code a Generator, based on what you learned building the Discriminator
      - what do we need to add for the Generator to do its job
      - GAN twist: tanh activation on output layer
      - exercise: how do weight initializations change network output from random input

    + live-coding demo, 20m: modifying our training loop for GAN
      - *instruction*: declaring a latent space, Pythonically, explicitly
      - demonstration: use functools.partial with torch.rand to create a "latent_space" callable
    + hands-on exercises, 10m: adding to your Generator, adding to the training loop
      - looking at initial outputs
      - plotting the loss after -- what does it look like? will compare to loss + discriminator outputs across training from GAN

- **Next steps with GANs: state-of-the-art models, 35m**
	-   *instruction*: conditional GANs, 10m
		-   compare and contrast with "unconditional GANs"
	-   *instruction*: DCGAN architecture, 10m
		-   instruction, 5m: what is a convolution
		-   *hands-on* examples, 5m: style transfer with pix2pix and CycleGAN
	-   alternate loss functions, 15m
		-   instruction: Wasserstein loss
			-   intuition: replace discriminator with critic that scores
		-  hands-on examples: rewrite Discriminator and training loop for original Wasserstein loss
-   *break*, 5m 
-   **GAN Applications**, 20m
	-   *instruction*, 5m: data augmentation
		-   hands-on exercise: 2D pottery GAN
	-   *instruction*, 5m: latent space visualization and exploration
		-   hands-on exercise: animal vocalization GANs
