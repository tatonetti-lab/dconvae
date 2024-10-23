import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse

# Custom Dataset for 4D Noisy Matrix
class NoisyMatrixDataset(Dataset):
    def __init__(self, original_matrix, noise_level=0.1, num_samples=1000):
        self.original_matrix = original_matrix
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.data = self.generate_noisy_samples()

    def generate_noisy_samples(self):
        noisy_samples = []
        for _ in range(self.num_samples):
            noisy_sample = self.original_matrix.clone()
            noise = torch.rand_like(noisy_sample)
            mask = noise < self.noise_level
            noisy_sample[mask] = 1 - noisy_sample[mask]  # Flip bits where mask is True (0 to 1 or 1 to 0)
            noisy_samples.append(noisy_sample)
        return torch.stack(noisy_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx].view(-1), self.original_matrix

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)       # Mean of latent variable
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)   # Log variance of latent variable

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(h))
        return x_reconstructed

# Define the VAE
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Loss function for VAE
def vae_loss(x, x_reconstructed, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence

# Main function to train the model
def train_model(initial_matrix, noise_level, num_samples, batch_size, hidden_dim, z_dim, epochs, lr, wd, device):
    dataset = NoisyMatrixDataset(initial_matrix, noise_level=noise_level, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = initial_matrix.numel()
    vae = VAE(input_dim, hidden_dim, z_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device).float()
            optimizer.zero_grad()
            x_reconstructed, mu, logvar = vae(data)
            loss = vae_loss(data, x_reconstructed, mu, logvar)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    print("Training complete.")

# Main script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a VAE on noisy 4D matrices.")
    
    # Command line arguments
    parser.add_argument('--noise_level', type=float, default=0.1, help="Noise level for generating samples.")
    parser.add_argument('--num_samples', type=int, default=10000, help="Number of noisy samples to generate.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--hidden_dim', type=int, default=400, help="Hidden layer dimension size.")
    parser.add_argument('--z_dim', type=int, default=20, help="Latent space dimension size.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--gpu_device', type=int, default=0, help="GPU device number (if available).")
    parser.add_argument('--wd', type=float, default=0, help="Weight decay for regularization.")

    args = parser.parse_args()

    # Generate initial 4D matrix
    initial_matrix = torch.randint(0, 2, (10, 4, 5, 6)).float()

    # Set device
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Train the model
    train_model(
        initial_matrix=initial_matrix,
        noise_level=args.noise_level,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        device=device
    )
