import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn.metrics
import torch
import pickle

def load_results(cv_list, base_path="experiments/results/mnist/quality"):
    base_path = Path(base_path)
    
    # Initialize arrays for metrics
    n_k = 50  # number of k values
    n_cv = len(cv_list)
    
    simplex_metrics = np.zeros((2, n_cv, n_k))
    knn_metrics = np.zeros((2, n_cv, n_k))
    
    # Load results for each CV fold
    for i, cv in enumerate(cv_list):
        # Load SimplEx results
        simplex_path = base_path / f"simplex_cv{cv}_n50.pkl"
        with open(simplex_path, 'rb') as f:
            simplex_data = pickle.load(f)
            
        # Load KNN results
        knn_path = base_path / f"nn_dist_cv{cv}_n50.pkl"
        with open(knn_path, 'rb') as f:
            knn_data = pickle.load(f)
            
        # Process results
        for k in range(n_k):
            # SimplEx metrics
            if 'latent_true' in simplex_data and 'latent_approx' in simplex_data:
                simplex_metrics[0, i, k] = sklearn.metrics.r2_score(
                    simplex_data['latent_true'][k], 
                    simplex_data['latent_approx'][k]
                )
            if 'output_true' in simplex_data and 'output_approx' in simplex_data:
                simplex_metrics[1, i, k] = sklearn.metrics.r2_score(
                    simplex_data['output_true'][k],
                    simplex_data['output_approx'][k]
                )
                
            # KNN metrics
            if 'latent_true' in knn_data and 'latent_approx' in knn_data:
                knn_metrics[0, i, k] = sklearn.metrics.r2_score(
                    knn_data['latent_true'][k],
                    knn_data['latent_approx'][k]
                )
            if 'output_true' in knn_data and 'output_approx' in knn_data:
                knn_metrics[1, i, k] = sklearn.metrics.r2_score(
                    knn_data['output_true'][k],
                    knn_data['output_approx'][k]
                )
    
    return simplex_metrics, knn_metrics

def plot_results(cv_list):
    # Load results
    simplex_metrics, knn_metrics = load_results(cv_list)
    
    # Plotting
    k_range = range(1, 51)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot latent space metrics
    mean_simplex_latent = np.mean(simplex_metrics[0], axis=0)
    std_simplex_latent = np.std(simplex_metrics[0], axis=0)
    mean_knn_latent = np.mean(knn_metrics[0], axis=0)
    std_knn_latent = np.std(knn_metrics[0], axis=0)
    
    ax1.plot(k_range, mean_simplex_latent, label='SimplEx')
    ax1.fill_between(k_range, 
                    mean_simplex_latent - std_simplex_latent,
                    mean_simplex_latent + std_simplex_latent,
                    alpha=0.3)
    ax1.plot(k_range, mean_knn_latent, label='KNN')
    ax1.fill_between(k_range,
                    mean_knn_latent - std_knn_latent,
                    mean_knn_latent + std_knn_latent,
                    alpha=0.3)
    
    ax1.set_xlabel('k')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Latent Space Approximation')
    ax1.legend()
    
    # Plot output space metrics
    mean_simplex_output = np.mean(simplex_metrics[1], axis=0)
    std_simplex_output = np.std(simplex_metrics[1], axis=0)
    mean_knn_output = np.mean(knn_metrics[1], axis=0)
    std_knn_output = np.std(knn_metrics[1], axis=0)
    
    ax2.plot(k_range, mean_simplex_output, label='SimplEx')
    ax2.fill_between(k_range,
                    mean_simplex_output - std_simplex_output,
                    mean_simplex_output + std_simplex_output,
                    alpha=0.3)
    ax2.plot(k_range, mean_knn_output, label='KNN')
    ax2.fill_between(k_range,
                    mean_knn_output - std_knn_output,
                    mean_knn_output + std_knn_output,
                    alpha=0.3)
    
    ax2.set_xlabel('k')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Output Space Approximation')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_approximation_quality.png')
    plt.close()

if __name__ == "__main__":
    # Use CV folds 0 and 1
    cv_list = [1, 2]
    plot_results(cv_list)
    print("Results plotted to mnist_approximation_quality.png")