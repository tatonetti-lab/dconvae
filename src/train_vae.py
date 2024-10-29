import os
import json
import argparse

import numpy as np
import scipy as sp

from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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

class ConfoundedDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise Exception(f"No dataset found at: {dataset_path}")

        config_path = os.path.join(dataset_path, 'config.json')
        if not os.path.exists(config_path):
            raise Exception(f"No config file found at: {config_path}")
        self.config = json.load(open(config_path))

        print("Loading drugs into tensor...")
        drugs = sp.sparse.load_npz(os.path.join(dataset_path, 'drugs.npz')).toarray()

        print("loading reactions (clean) into tensor...")
        reactions = sp.sparse.load_npz(os.path.join(dataset_path, 'reactions.npz')).toarray()

        print("Creating empty indications matrix to use as target.")
        indications = np.zeros(shape=(self.config['nreports'], self.config['nindications']))

        self.target = torch.tensor(np.hstack([drugs, reactions, indications]), dtype=torch.float)
        
        example_keys = sorted([key for key in self.config.keys() if key.startswith('dataset')])
        self.num_examples = len(example_keys)

        print(f"Found dataset with {self.num_examples} examples of confounding.")

        confounded_examples = list()

        print("Loading examples...")

        for i, key in tqdm(enumerate(example_keys)):

            reactions_observed = sp.sparse.load_npz(os.path.join(dataset_path, 'datasets', f'{i}_reactions_observed.npz')).toarray()
            indications_observed = sp.sparse.load_npz(os.path.join(dataset_path, 'datasets', f'{i}_indications.npz')).toarray()

            data = torch.tensor(np.hstack([drugs, reactions_observed, indications_observed]), dtype=torch.float)

            confounded_examples.append(data)
        
        self.confounded_examples = torch.stack(confounded_examples)
        
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.confounded_examples[idx].view(-1), self.target

# Custom Dataset for 4D Noisy Matrix
class NoisyMatrixDataset(Dataset):
    def __init__(self, original_matrix, noise_level=0.1, num_samples=1000):
        self.target = original_matrix
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.data = self.generate_noisy_samples()

    def generate_noisy_samples(self):
        noisy_samples = []
        for _ in range(self.num_samples):
            noisy_sample = self.target.clone()
            noise = torch.rand_like(noisy_sample)
            mask = noise < self.noise_level
            noisy_sample[mask] = 1 - noisy_sample[mask]  # Flip bits where mask is True (0 to 1 or 1 to 0)
            noisy_samples.append(noisy_sample)
        return torch.stack(noisy_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx].view(-1), self.target

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
def train_model_prev(dataset, dataloader, hidden_dim, z_dim, epochs, lr, wd, device, save_dir='outputs'):
    
    input_dim = dataset.target.numel()
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

def train_model(dataset, dataloader, hidden_dim, z_dim, epochs, lr, wd, device, save_dir='outputs'):
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    input_dim = dataset.target.numel()
    vae = VAE(input_dim, hidden_dim, z_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)

    # Lists to track metrics
    training_history = {
        'losses': [],
        'accuracies': []
    }

    best_accuracy = 0
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device).float()
            optimizer.zero_grad()

            x_reconstructed, mu, logvar = vae(data)
            loss = vae_loss(data, x_reconstructed, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_accuracy = compute_accuracy(x_reconstructed, data)
            total_accuracy += batch_accuracy
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        # Save metrics
        training_history['losses'].append(avg_loss)
        training_history['accuracies'].append(avg_accuracy)

        # # Save best model
        # TODO: This is super inefficeitn and slows down training. Will need to come up with a 
        # TODO: different way to accomplish this. 
        # if avg_accuracy > best_accuracy:
        #     best_accuracy = avg_accuracy
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': vae.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_loss,
        #         'accuracy': avg_accuracy
        #     }, os.path.join(save_dir, 'best_model.pt'))

        if (epoch % int(epochs/10)) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

    # Save final model
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }, os.path.join(save_dir, 'final_model.pt'))

    # Generate and save final predictions
    vae.eval()
    with torch.no_grad():
        # Create a tensor to store all predictions
        all_predictions = []
        
        # Process each batch
        for batch_data, _ in dataloader:
            batch_data = batch_data.to(device).float()
            reconstructed, _, _ = vae(batch_data)
            binary_predictions = torch.round(reconstructed)
            all_predictions.append(binary_predictions.cpu())
        
        # Concatenate all predictions
        final_predictions = torch.cat(all_predictions, dim=0)
        
        # Save final predictions
        #torch.save(final_predictions, os.path.join(save_dir, 'final_predictions.pt'))
        np.save(os.path.join(save_dir, 'final_predictions.npy'), final_predictions.numpy())

    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save final performance metrics
    final_metrics = {
        'final_loss': training_history['losses'][-1],
        'final_accuracy': training_history['accuracies'][-1],
        'best_accuracy': best_accuracy,
        'training_epochs': epochs
    }
    
    with open(os.path.join(save_dir, 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print("\nTraining complete. Saved:")
    print(f"- Best and final models in {save_dir}")
    print(f"- Final predictions in {save_dir}")
    print(f"- Training history and metrics in {save_dir}")
    
    return vae, final_metrics

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
    parser.add_argument('--dataset', type=str, default=None, help="Path to the dataset to use. Default is to generate a random noise example.")
    parser.add_argument('--save_dir', type=str, default='outputs', help="Directory to save models and results")
    
    args = parser.parse_args()

    if args.dataset is None:
        
        print("No dataset provided. Will generate a dataset with random noise.")

        nreports = 100
        ndrugs = 10
        nreactions = 11
        nindications = 12

        drugs = np.random.binomial(1, args.sparsity, size=(nreports, ndrugs))
        reactions = np.random.binomial(1, args.sparsity, size=(nreports, nreactions))
        indications = np.random.binomial(1, args.sparsity, size=(nreports, nindications))
        data = np.hstack([drugs, reactions, indications])
        print(data.shape)

        initial_matrix = torch.tensor(data, dtype=torch.float)

        dataset = NoisyMatrixDataset(initial_matrix, noise_level=args.noise_level, num_samples=args.num_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print(f"Loading dataset from {args.dataset}")
        dataset = ConfoundedDataset(args.dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset_name = args.dataset.split('/')[-1]
        print(f"Dataset name: {dataset_name}")

        if args.save_dir == 'outputs':
            args.save_dir = os.path.join('outputs', dataset_name)
        
        print(f"Will save the output to: {args.save_dir}")

    # Initialize a tensor to accumulate the sum of all the samples
    sum_matrix = torch.zeros_like(dataset.target)

    # Loop through the dataset and sum all examples
    for data, _ in dataloader:
        sum_matrix += data.sum(dim=0).view(dataset.target.size())

    # Compute the average by dividing the sum by the number of samples
    average_matrix = sum_matrix / len(dataset)

    # Convert the average matrix to binary by rounding
    binary_average_matrix = torch.round(average_matrix)

    # print("Average non-zero matrix positions in noisy data:")
    # for idx in torch.nonzero(binary_average_matrix, as_tuple=False):
    #     print(idx.tolist())

    print(f"Accuracy for averaged noisy dataset: {compute_accuracy(binary_average_matrix, dataset.target)}")

    # Set device
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Train the model
    model, final_metrics = train_model(
        dataset=dataset,
        dataloader=dataloader,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        device=device,
        save_dir=args.save_dir
    )

    print("\nFinal Performance Metrics:")
    print(f"Final Loss: {final_metrics['final_loss']:.4f}")
    print(f"Final Accuracy: {final_metrics['final_accuracy']:.2f}%")
    print(f"Best Accuracy: {final_metrics['best_accuracy']:.2f}%")
