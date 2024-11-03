import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import inception_v3
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from scipy import linalg
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define generator model (unchanged)
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 8*8*256),
            nn.BatchNorm1d(8*8*256),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 8, 8)),
            
            nn.ConvTranspose2d(256, 128, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(128, 64, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 32, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(32, 3, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Define discriminator model (adjusted)
class Discriminator(nn.Module):
    def __init__(self, input_channels=3, input_height=128, input_width=128):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Conv2d(128, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Flatten()
        )
        
        # Calculate the size of the output from convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.main(dummy_input)
            self.conv_out_size = conv_out.view(1, -1).size(1)
        
        self.fc = nn.Linear(self.conv_out_size, 1)
    
    def forward(self, x):
        x = self.main(x)
        return self.fc(x.view(x.size(0), -1))

# Loss functions (unchanged)
def discriminator_loss(real_output, fake_output):
    real_loss = nn.BCEWithLogitsLoss()(real_output, torch.ones_like(real_output))
    fake_loss = nn.BCEWithLogitsLoss()(fake_output, torch.zeros_like(fake_output))
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return nn.BCEWithLogitsLoss()(fake_output, torch.ones_like(fake_output))


# Function to calculate FID score
def calculate_fid(real_images, fake_images, device, batch_size=64):
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    def get_activations(images):
        activations = []
        with torch.no_grad():
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i+batch_size]
                batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                batch = batch.to(device)
                pred = inception_model(batch)
                activations.append(pred.cpu().numpy())
        return np.concatenate(activations, axis=0)

    real_activations = get_activations(real_images)
    fake_activations = get_activations(fake_images)

    mu_real, sigma_real = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu_fake, sigma_fake = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2*covmean)
    return fid


# Modified training step
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim):
    batch_size = images.size(0)
    noise = torch.randn(batch_size, noise_dim, device=images.device)
    
    # Train Generator
    generator_optimizer.zero_grad()
    generated_images = generator(noise)
    fake_output = discriminator(generated_images)
    gen_loss = generator_loss(fake_output)
    gen_loss.backward()
    generator_optimizer.step()
    
    # Train Discriminator
    discriminator_optimizer.zero_grad()
    real_output = discriminator(images)
    fake_output = discriminator(generated_images.detach())
    disc_loss = discriminator_loss(real_output, fake_output)
    disc_loss.backward()
    discriminator_optimizer.step()
    
    return gen_loss.item(), disc_loss.item()

# Modified generate_and_save_images function
def generate_and_save_images(model, epoch, test_input, real_images, device):
    model.eval()
    with torch.no_grad():
        predictions = model(test_input)
    model.train()
    
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.size(0)):
        plt.subplot(4, 4, i+1)
        plt.imshow(((predictions[i].cpu() * 0.5 + 0.5) * 255).permute(1, 2, 0).byte())
        plt.axis('off')
    
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()

    # Calculate FID score
    fid_score = calculate_fid(real_images, predictions, device)
    print(f"FID Score at epoch {epoch}: {fid_score}")
    return fid_score


# Load and preprocess PlantVillage dataset (adjusted)
def load_image_dataset(directory, batch_size):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Ensure consistent size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = datasets.ImageFolder(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

# Modified training loop
def train(dataloader, epochs, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, gen_scheduler, disc_scheduler, device):
    seed = torch.randn(16, noise_dim, device=device)
    fid_scores = []
    
    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, noise_dim)
                pbar.update(1)
                pbar.set_postfix({"Gen Loss": f"{gen_loss:.4f}", "Disc Loss": f"{disc_loss:.4f}"})

        if (epoch + 1) % 5 == 0:
            fid_score = generate_and_save_images(generator, epoch + 1, seed, image_batch, device)
            fid_scores.append(fid_score)
            # In your training loop, after calculating the FID score:
            gen_scheduler.step(fid_score)
            disc_scheduler.step(fid_score)


        elapsed_time = time.time() - start_time
        estimated_time = (elapsed_time / (epoch + 1)) * epochs
        remaining_time = estimated_time - elapsed_time
        
        
        print(f"Estimated time remaining: {remaining_time/60:.2f} minutes")
        print(f"Current learning rates - Generator: {gen_scheduler.optimizer.param_groups[0]['lr']:.6f}, Discriminator: {disc_scheduler.optimizer.param_groups[0]['lr']:.6f}")

    # Plot FID scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(5, epochs + 1, 5), fid_scores)
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Score over Training')
    plt.savefig('fid_scores.png')
    plt.close()


# Modified main execution
def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 1000
    noise_dim = 100
    initial_lr = 1e-4

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataloader = load_image_dataset('/kaggle/input/grape-plant-from-plant-village-dataset/Grape Plant from Plant Village Dataset', BATCH_SIZE)

    # Get input dimensions from a sample batch
    sample_batch, _ = next(iter(dataloader))
    input_channels, input_height, input_width = sample_batch.shape[1:]

    # Build models
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator(input_channels, input_height, input_width).to(device)

    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=initial_lr, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=initial_lr, betas=(0.5, 0.999))

    # Learning rate schedulers
    gen_scheduler = ReduceLROnPlateau(generator_optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)
    disc_scheduler = ReduceLROnPlateau(discriminator_optimizer, 'min', factor=0.5, patience=10, min_lr=1e-6)

    # Train
    train(dataloader, EPOCHS, noise_dim, generator, discriminator, generator_optimizer, discriminator_optimizer, gen_scheduler, disc_scheduler, device)

if __name__ == "__main__":
    main()
