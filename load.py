import torch
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class DataHandler:
    @staticmethod
    def get_mnist_data(data_dir='./data'):
        """Load MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download training data
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )
        
        # Download test data
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset
    
    @staticmethod
    def get_prostate_cancer_data():
        """
        Download prostate cancer dataset from SEER
        Note: This is a simulated version for demonstration
        """
        # Generate synthetic data for demonstration
        n_samples = 1000
        np.random.seed(42)
        
        data = {
            'age': np.random.normal(65, 10, n_samples),
            'psa': np.random.lognormal(2, 1, n_samples),
            'gleason_score': np.random.choice([6, 7, 8, 9], n_samples),
            'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_samples),
            'survival_months': np.random.exponential(60, n_samples)
        }
        
        df = pd.DataFrame(data)
        df['outcome'] = (df['survival_months'] < 36).astype(int)
        
        return df

class ProstateDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        
        # Convert categorical variables
        self.stage_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        self.data['stage_encoded'] = self.data['stage'].map(self.stage_map)
        
        # Normalize numerical features
        numerical_cols = ['age', 'psa', 'gleason_score']
        self.means = self.data[numerical_cols].mean()
        self.stds = self.data[numerical_cols].std()
        self.data[numerical_cols] = (self.data[numerical_cols] - self.means) / self.stds
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        features = torch.tensor([
            self.data.iloc[idx]['age'],
            self.data.iloc[idx]['psa'],
            self.data.iloc[idx]['gleason_score'],
            self.data.iloc[idx]['stage_encoded']
        ], dtype=torch.float32)
        
        label = torch.tensor(self.data.iloc[idx]['outcome'], dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
            
        return features, label

def prepare_datasets():
    """Prepare both MNIST and Prostate Cancer datasets"""
    # Get MNIST
    mnist_train, mnist_test = DataHandler.get_mnist_data()
    
    # Get Prostate Cancer data
    prostate_df = DataHandler.get_prostate_cancer_data()
    train_df, test_df = train_test_split(prostate_df, test_size=0.2, random_state=42)
    
    prostate_train = ProstateDataset(train_df)
    prostate_test = ProstateDataset(test_df)
    
    return {
        'mnist': (mnist_train, mnist_test),
        'prostate': (prostate_train, prostate_test)
    }

# Example usage
if __name__ == "__main__":
    # Load all datasets
    datasets = prepare_datasets()
    
    # Create dataloaders
    mnist_train_loader = DataLoader(
        datasets['mnist'][0],
        batch_size=32,
        shuffle=True
    )
    
    prostate_train_loader = DataLoader(
        datasets['prostate'][0],
        batch_size=32,
        shuffle=True
    )
    
    # Print dataset sizes
    print("MNIST Training samples:", len(datasets['mnist'][0]))
    print("MNIST Test samples:", len(datasets['mnist'][1]))
    print("Prostate Cancer Training samples:", len(datasets['prostate'][0]))
    print("Prostate Cancer Test samples:", len(datasets['prostate'][1]))
    
    # Get a sample batch
    mnist_features, mnist_labels = next(iter(mnist_train_loader))
    prostate_features, prostate_labels = next(iter(prostate_train_loader))
    
    print("\nMNIST feature shape:", mnist_features.shape)
    print("Prostate Cancer feature shape:", prostate_features.shape)