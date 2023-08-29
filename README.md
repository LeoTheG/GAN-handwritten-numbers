# Generative Adversarial Networks

## Training

`python3 src/train_model.py`

Will create a model `generator.pth`

## Running

`python3 src/main.py`

Will generate a collection of 8 images in `"fake_images.png"`

## Explanation

A Generative Adversarial Network (GAN) is a type of deep learning model where two neural networks, the Generator and the Discriminator, are trained together. The Generator tries to produce fake data, while the Discriminator attempts to distinguish between real and fake data. Through this adversarial process, the Generator improves its ability to create data that looks real, leading to the generation of synthetic data that closely resembles genuine data.

The number of epochs needed to generate "usable" images with a Generative Adversarial Network (GAN) can vary widely based on:

The complexity of the dataset: Simpler datasets might converge faster than more complex ones.
The architecture of the GAN: Certain architectures might require more or fewer epochs to generate decent images.
The training procedure: Things like learning rate, batch size, and optimizers can influence the convergence.
The quality of the initialization and randomness involved in training.
For simple datasets like MNIST (handwritten digits), good results can often be seen in as little as a few thousand epochs. For more complex datasets like CIFAR-10, you might need tens of thousands of epochs. For high-resolution and very complex datasets (e.g., generating faces or artwork), training might require hundreds of thousands to millions of epochs.

A good approach is to periodically visualize the output of your generator as it trains. This way, you can monitor its progress and make an informed decision about when to stop training or if adjustments need to be made.

In practice, it's common to save the model's state at regular intervals, say every 1,000 epochs, and manually inspect the quality of generated samples. This approach allows for both automated and human-in-the-loop evaluation of the model's progress.
