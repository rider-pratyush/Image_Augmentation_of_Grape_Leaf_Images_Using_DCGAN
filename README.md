## Image Augmentation of Grape Leaf Images using DCGAN
Deep learning models for leaf disease identification often suffer from limited training data, which leads to reduced performance. This paper explores the use of a Deep Convolutional Generative Adversarial Network (DCGAN) for data
augmentation of grape leaf images to address this challenge. A DCGANmodel, based on the original DCGAN paper, was trained on a dataset of 4,062 authentic grape leaf images to generate16 synthetic samples. 
The quality of these generated images was assessed using the Fr´echet Inception Distance (FID) score. 
Theimpact of this limited data augmentation on classification accuracy was evaluated using a pre-trained Inception V3 model from PyTorch. The paper aims to highlight the results of DCGAN-
based data augmentation for grape leaf disease identification, even with a limited number of generated images. The findings contribute to understanding the feasibility and effectiveness of
this approach for improving classification accuracy in scenarios with constrained resources.
## About the Model
In this model, both the generator and discriminator improve by playing a "min-max" game where the generator tries to create realistic images, and the discriminator becomes better at spotting fake ones.
