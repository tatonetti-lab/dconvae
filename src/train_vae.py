import os
import json
import argparse

import numpy as np
import scipy as sp

from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
    def __init__(self, dataset_paths):
        if type(dataset_paths) == str:
            dataset_paths = [dataset_paths]
        
        self.num_examples = 0
        self.targets = list()
        self.example2target = list()
        confounded_examples = list()
        self.configs = list()
        self.dataset_paths = dataset_paths

        for dataset_idx, dataset_path in enumerate(dataset_paths):
            
            if not os.path.exists(dataset_path):
                raise Exception(f"No dataset found at: {dataset_path}")

            config_path = os.path.join(dataset_path, 'config.json')
            if not os.path.exists(config_path):
                raise Exception(f"No config file found at: {config_path}")
            config = json.load(open(config_path))
            self.configs.append( config )

            print("Loading drugs into tensor...")
            drugs = sp.sparse.load_npz(os.path.join(dataset_path, 'drugs.npz')).toarray()

            print("loading reactions (clean) into tensor...")
            reactions = sp.sparse.load_npz(os.path.join(dataset_path, 'reactions.npz')).toarray()
            
            print("initiatlizing empty matrix to stand in for indications...")
            indications = np.zeros(shape=(config['nreports'], config['nindications']))

            self.targets.append(torch.tensor(np.hstack([reactions, drugs, indications]), dtype=torch.float))
            
            example_keys = sorted([key for key in config.keys() if key.startswith('dataset')])
            self.num_examples += len(example_keys)

            print(f"Found dataset with {len(example_keys)} examples of confounding.")

            print("Loading examples...")

            for i, key in tqdm(enumerate(example_keys)):

                reactions_observed = sp.sparse.load_npz(os.path.join(dataset_path, 'datasets', f'{i}_reactions_observed.npz')).toarray()
                indications_observed = sp.sparse.load_npz(os.path.join(dataset_path, 'datasets', f'{i}_indications.npz')).toarray()

                data = torch.tensor(np.hstack([reactions_observed, drugs, indications_observed]), dtype=torch.float)

                confounded_examples.append(data)
                self.example2target.append(dataset_idx)
        
        # nreactions = set([c['nreactions'] for c in self.configs])
        # if len(nreactions) != 1:
        #     raise Exception(f"ERROR: Datasets being combined have a different number of reactions. Must be the same.")
        # self.nreactions = list(nreactions)[0]
        
        self.input_dim = confounded_examples[0].numel()
        self.output_dim = self.targets[0].numel()
        self.confounded_examples = torch.stack(confounded_examples)
    
    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return self.confounded_examples[idx].view(-1), self.targets[self.example2target[idx]].view(-1)

    def save_predictions(self, predictions, name, dirpath):
        """
        Saves the predictions to match up with the Dataset which can be composed of multiple simulated
        datasets.
        """
        predictions = predictions.numpy()

        start_index = 0
        end_index = 0
        for dataset_idx, dataset_path in enumerate(self.dataset_paths):
            
            if not os.path.exists(dataset_path):
                raise Exception(f"No dataset found at: {dataset_path}")

            config_path = os.path.join(dataset_path, 'config.json')
            if not os.path.exists(config_path):
                raise Exception(f"No config file found at: {config_path}")
            config = json.load(open(config_path))

            example_keys = sorted([key for key in config.keys() if key.startswith('dataset')])

            end_index += len(example_keys)

            np.save(os.path.join(dirpath, f'final_predictions_{name}_{dataset_idx}'), predictions[start_index:end_index])

            start_index = end_index

# Custom Dataset for 4D Noisy Matrix
class NoisyMatrixDataset(Dataset):
    def __init__(self, original_matrix, noise_level=0.1, num_samples=1000):
        self.target = original_matrix
        self.noise_level = noise_level
        self.num_samples = num_samples
        # self.nreactions = original_matrix.shape[1] # for compatability with ConfoundedDataset definition
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

    # def save_predictions(self, predictions, name, filepath):
    #     print(f"Save on NoisyMatrixDataset is not implemented.")

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
    def __init__(self, input_dim, hidden_dim, z_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, output_dim)

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

# weighted loss functions
def weighted_vae_loss(x, x_reconstructed, mu, logvar, pos_weight=None):
    """
    VAE loss function with weighted binary cross entropy for sparse data.
    
    Args:
        x: Original input tensor
        x_reconstructed: Reconstructed input tensor
        mu: Mean tensor from encoder
        logvar: Log variance tensor from encoder
        pos_weight: Weight for positive cases (1s). Can be a single value or tensor
                   matching the dimensions of x
    """
    if pos_weight is None:
        # Automatically calculate weight based on sparsity
        num_ones = torch.sum(x == 1)
        num_zeros = torch.sum(x == 0)
        pos_weight = num_zeros / num_ones if num_ones > 0 else 1.0
        pos_weight = torch.tensor(pos_weight, device=x.device)

    # Using BCE with logits for numerical stability
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        x_reconstructed,
        x,
        pos_weight=pos_weight,
        reduction='sum'
    )
    
    # KL divergence term remains the same
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
    # [:,nreactions]
    # print(reconstructed_matrix.shape)
    
    # Check if the dimensions of both matrices are the same
    assert reconstructed_matrix.size() == initial_matrix.size(), "Matrices must have the same dimensions."
    
    # Compare the two matrices element-wise and calculate the number of matches
    # correct_predictions = torch.eq(reconstructed_matrix, initial_matrix).sum().item()
    correct_predictions = torch.eq(reconstructed_matrix[initial_matrix==1], initial_matrix[initial_matrix==1]).sum().item()
    
    # Calculate the total number of elements
    # total_elements = initial_matrix.numel()
    total_elements = initial_matrix[initial_matrix==1].sum().item()
    
    # Compute accuracy as the percentage of correct predictions
    accuracy = (correct_predictions / total_elements) * 100
    
    return accuracy

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a given dataloader.
    
    Args:
        model: VAE model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        tuple: (average loss, average accuracy)
    """
    model.eval()
    total_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device).float()
            target = target.to(device).float()
            x_reconstructed, mu, logvar = model(data)
            loss = weighted_vae_loss(target, x_reconstructed, mu, logvar)
            
            total_loss += loss.item()
            batch_accuracy = compute_accuracy(x_reconstructed, target)
            total_accuracy += batch_accuracy
    
    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    
    model.train()
    return avg_loss, avg_accuracy

def train_model(dataloader, val_dataloader, hidden_dim, z_dim, epochs, lr, wd, device, save_dir='outputs'):
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    input_dim = dataloader.dataset.input_dim
    output_dim = dataloader.dataset.output_dim
    
    vae = VAE(input_dim, hidden_dim, z_dim, output_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=wd)

    # Lists to track metrics
    training_history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': []
    }

    best_accuracy = 0
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device).float()
            target = target.to(device).float()
            optimizer.zero_grad()

            x_reconstructed, mu, logvar = vae(data)
            loss = weighted_vae_loss(target, x_reconstructed, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_accuracy = compute_accuracy(x_reconstructed, target)
            total_accuracy += batch_accuracy
        
        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)
        
        # validation metrics
        val_loss, val_accuracy = evaluate_model(vae, val_dataloader, device)

        # Save metrics
        training_history['train_losses'].append(avg_loss)
        training_history['train_accuracies'].append(avg_accuracy)
        training_history['val_losses'].append(val_loss)
        training_history['val_accuracies'].append(val_accuracy)

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

        if True or (epoch % int(epochs/10)) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Acc: {avg_accuracy:.2f}%, Val Loss: {val_loss:.2f}, Val Acc: {val_accuracy:.2f}')

    # Save final model
    torch.save({
        'epoch': epochs-1,
        'model_state_dict': vae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_loss,
        'train_accuracy': avg_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    }, os.path.join(save_dir, 'final_model.pt'))

    # Generate and save final predictions
    vae.eval()
    with torch.no_grad():
        for split, dl in [('train', dataloader), ('val', val_dataloader)]:
            # Create a tensor to store all predictions
            all_predictions = []
            
            # Process each batch
            for batch_data, target in dl:
                batch_data = batch_data.to(device).float()
                target = target.to(device).float()
                reconstructed, _, _ = vae(batch_data)
                binary_predictions = torch.round(reconstructed)
                all_predictions.append(binary_predictions.cpu())
            
            # Concatenate all predictions
            final_predictions = torch.cat(all_predictions, dim=0)
            
            # Save final predictions
            #torch.save(final_predictions, os.path.join(save_dir, 'final_predictions.pt'))
            np.save(os.path.join(save_dir, f'final_predictions_{split}'), final_predictions)
            dl.dataset.save_predictions(final_predictions, split, save_dir)

    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save final performance metrics
    final_metrics = {
        'final_train_loss': training_history['train_losses'][-1],
        'final_train_accuracy': training_history['train_accuracies'][-1],
        'final_val_loss': training_history['val_losses'][-1],
        'final_val_accuracy': training_history['val_accuracies'][-1],
        #'best_accuracy': best_accuracy,
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
    parser.add_argument('--num-samples', type=int, default=10000, help="Number of noisy samples to generate.")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--hidden-dim', type=int, default=400, help="Hidden layer dimension size.")
    parser.add_argument('--z-dim', type=int, default=20, help="Latent space dimension size.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device number (if available).")
    parser.add_argument('--wd', type=float, default=0, help="Weight decay for regularization.")
    parser.add_argument('--sparsity', type=float, default=0.95, help="Sparsity of the randomly initialized matrix. Default is 0.95, 95% of the matrix will be 0's")
    
    parser.add_argument('--dataset', nargs='+', type=str, default=None, help="Path to the dataset to use. Default is to generate a random noise example.")
    parser.add_argument('--save-dir', type=str, default='outputs', help="Directory to save models and results")
    parser.add_argument('--val-dataset', type=str, default=None, help="Path to the validation dataset to use. Default is None.")

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

        val_dataset = None
        val_dataloader = None

        if args.save_dir == 'outputs':
            args.save_dir = os.path.join('outputs', 'random_noise', f'{nreports}_{ndrugs}_{nreactions}_{nreactions}_{args.sparsity}_{args.noise_level}_{args.num_samples}')
        skip_args = ['val_dataset', 'dataset']

    else:
        print(f"Loading datasets from {args.dataset}")
        dataset = ConfoundedDataset(args.dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        dataset_name = '_'.join([dataset.split('/')[-1] for dataset in args.dataset])
        print(f"Dataset name: {dataset_name}")

        if args.save_dir == 'outputs':
            output_dirname = dataset_name
            if args.val_dataset is not None:
                val_dataset_name = args.val_dataset.split('/')[-1]
                output_dirname += f"_{val_dataset_name}"
            
            args.save_dir = os.path.join('outputs', output_dirname)
        
        print(f"Will save the output to: {args.save_dir}")

        if args.val_dataset is not None:
            print(f"    Loading validation dataset from {args.val_dataset}")
            val_dataset = ConfoundedDataset(args.val_dataset)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        skip_args = ['noise_level', 'num_samples', 'sparsity']

    # # Initialize a tensor to accumulate the sum of all the samples
    # sum_matrix = torch.zeros_like(dataset.targets[0])

    # # Loop through the dataset and sum all examples
    # for data, _ in dataloader:
    #     sum_matrix += data.sum(dim=0).view(dataset.targets[0].size())

    # # Compute the average by dividing the sum by the number of samples
    # average_matrix = sum_matrix / len(dataset)

    # # Convert the average matrix to binary by rounding
    # binary_average_matrix = torch.round(average_matrix)

    # # print("Average non-zero matrix positions in noisy data:")
    # # for idx in torch.nonzero(binary_average_matrix, as_tuple=False):
    # #     print(idx.tolist())

    # print(f"Accuracy for averaged noisy dataset: {compute_accuracy(binary_average_matrix, dataset.target)}")

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    # Train the model
    model, final_metrics = train_model(
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        device=device,
        save_dir=args.save_dir
    )

    print("\nFinal Performance Metrics:")
    print(f"Final Loss: {final_metrics['final_train_loss']:.4f}")
    print(f"Final Accuracy: {final_metrics['final_train_accuracy']:.2f}%")
    print(f"Final Val Loss: {final_metrics['final_val_loss']:.4f}")
    print(f"Final Val Accuracy: {final_metrics['final_val_accuracy']:.2f}%")


    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        save_args = dict()
        for key, val in args.__dict__.items():
            if not key in skip_args:
                save_args[key] = val
        json.dump(save_args, f, indent=2)
    