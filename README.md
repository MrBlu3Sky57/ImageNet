# ImageNet
An implementation of a CNN in NumPy and PyTorch for image classification

## Table of Contents
- [Plan](#plan)
- [Implementation](#implementation)
- [Takeaways](#takeaways)
- [License](#license)

## Plan
In this project I will build up the mathematical formulations of a Convolutional Neural Network
(See the writeups folder) then use this theory to build a image classifier

## Implementation
I discuss the mathematical details of my training and overall model design
in my architecture write up. The main idea of this project was to get an understanding of Convolutional Neural Networks from the ground up, using NumPy to build the model from scratch. I also built a PyTorch version to familiarize myself with the framework, and saw similar performance on the CPU compared to my model. However, GPU acceleration increased training speed significantly.

## Takeaways
I was quite proud of the results of my model, my from scratch CNN had 98.3 accuracy on MNIST with pretty much no hyperparameter tuning, and all of my implementation
done in pure NumPy. My PyTorch implementation was able to reach 98.8 accuracy, but mostly because I was able to train it for much longer.
