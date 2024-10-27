import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse

def initialize_sparse_matrix(shape, sparsity):
    """
    Initialize a sparse binary matrix with a given sparsity level.
    
    Args:
    - shape (tuple): The shape of the matrix (e.g., (10, 4, 5, 4)).
    - sparsity (float): The desired sparsity level, where 1.0 means fully sparse (all zeros),
                        and 0.0 means fully dense (all ones).
    
    Returns:
    - torch.Tensor: A randomly initialized sparse binary matrix with the given sparsity.
    """
    assert 0.0 <= sparsity <= 1.0, "Sparsity must be between 0 and 1."

    # Calculate the number of elements that should be set to 1 based on sparsity
    total_elements = torch.prod(torch.tensor(shape)).item()
    num_ones = int(total_elements * (1 - sparsity))  # Elements to set to 1

    # Create a flat tensor of zeros
    matrix = torch.zeros(total_elements)

    # Randomly select positions to set to 1
    indices = torch.randperm(total_elements)[:num_ones]
    matrix[indices] = 1

    # Reshape the flat tensor back into the desired shape
    matrix = matrix.view(*shape)

    return matrix

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

def compute_accuracy(reconstructed_matrix, initial_matrix):
    """
    Computes the accuracy of the reconstructed matrix compared to the initial matrix.
    
    Args:
    - reconstructed_matrix (torch.Tensor): The reconstructed matrix (binary).
    - initial_matrix (torch.Tensor): The initial matrix (binary).
    
    Returns:
    - accuracy (float): The accuracy of the reconstructed matrix as a percentage.
    """
    # Ensure the matrices are binary
    reconstructed_matrix = torch.round(reconstructed_matrix)
    
    # Check if the dimensions of both matrices are the same
    assert reconstructed_matrix.size() == initial_matrix.size(), "Matrices must have the same dimensions."
    
    # Compare the two matrices element-wise and calculate the number of matches
    correct_predictions = torch.eq(reconstructed_matrix, initial_matrix).sum().item()
    
    # Calculate the total number of elements
    total_elements = initial_matrix.numel()
    
    # Compute accuracy as the percentage of correct predictions
    accuracy = (correct_predictions / total_elements) * 100
    
    return accuracy

# Main function to train the model
def train_model(initial_matrix, noise_level, num_samples, batch_size, hidden_dim, z_dim, epochs, lr, wd, device):
    
    print("Initial non-zero matrix positions:")
    for idx in torch.nonzero(initial_matrix, as_tuple=False):
        print(idx.tolist())
    
    dataset = NoisyMatrixDataset(initial_matrix, noise_level=noise_level, num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize a tensor to accumulate the sum of all the samples
    sum_matrix = torch.zeros_like(initial_matrix)

    # Loop through the dataset and sum all examples
    for data, _ in dataloader:
        sum_matrix += data.sum(dim=0).view(initial_matrix.size())

    # Compute the average by dividing the sum by the number of samples
    average_matrix = sum_matrix / num_samples

    # Convert the average matrix to binary by rounding
    binary_average_matrix = torch.round(average_matrix)

    # print("Average non-zero matrix positions in noisy data:")
    # for idx in torch.nonzero(binary_average_matrix, as_tuple=False):
    #     print(idx.tolist())

    print(f"Accuracy for averaged noisy dataset: {compute_accuracy(binary_average_matrix, initial_matrix)}")
    
    input_dim = initial_matrix.numel()
    vae = VAE(input_dim, hidden_dim, z_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device).float()  # Convert to float
            optimizer.zero_grad()

            # Forward pass through the VAE
            x_reconstructed, mu, logvar = vae(data)

            # Compute loss and backpropagate
            loss = vae_loss(data, x_reconstructed, mu, logvar)
            loss.backward()
            optimizer.step()

            # Accumulate total loss
            total_loss += loss.item()

            # Compute accuracy for this batch
            batch_accuracy = compute_accuracy(x_reconstructed, data)
            total_accuracy += batch_accuracy
        
        # Calculate average loss and accuracy for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        # Print the results for the epoch
        if (epoch % int(epochs/10)) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

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
    parser.add_argument('--sparsity', type=float, default=0.95, help="Sparsity of the randomly initialized matrix. Default is 0.95, 95% of the matrix will be 0's")

    args = parser.parse_args()

    # Generate initial 4D matrix
    # Reports x Drugs x Indications x Side Effects
    # 0         0       0             0
    # 1         1       0             1
    # 2         1       0             1
    # 3         2       0             2
    # 4         2       0             2
    # 5         3       0             3
    # 6         3       0             3
    # 7         3       0             3
    # 8         0       0             0
    # 9         0       0             0
    # positions = [
    #     [0, 0, 0, 0],
    #     [1, 1, 0, 1],
    #     [2, 1, 0, 1],
    #     [3, 2, 0, 2],
    #     [4, 2, 0, 2],
    #     [5, 3, 0, 3],
    #     [6, 3, 0, 3],
    #     [7, 3, 0, 3],
    #     [8, 0, 0, 0],
    #     [9, 0, 0, 0]
    # ]
    # initial_matrix = torch.zeros((10, 4, 5, 6))
    # for pos in positions:
    #     initial_matrix[pos[0], pos[1], pos[2], pos[3]] = 1
    
    shape = (10, 4, 5, 4)  # Shape of the 4D matrix
    
    initial_matrix = initialize_sparse_matrix(shape, args.sparsity)

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
