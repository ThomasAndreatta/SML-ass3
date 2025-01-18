import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn.metrics
import torch

def calculate_metrics(simplex_obj):
    """Calculate metrics from SimplEx object"""
    # Get latent representations
    test_latents = simplex_obj.test_latent_reps
    
    # Compute approximation using weights and corpus
    latent_approx = torch.matmul(simplex_obj.weights, simplex_obj.corpus_latent_reps)
    
    # Calculate R² scores for different k values
    k_values = range(1, 51)  # From 1 to 50 neighbors
    latent_scores = []
    
    for k in k_values:
        # Sort weights and get top k for each test example
        top_k_weights = torch.zeros_like(simplex_obj.weights)
        for i in range(len(test_latents)):
            _, indices = torch.topk(simplex_obj.weights[i], k)
            top_k_weights[i, indices] = simplex_obj.weights[i, indices]
        
        # Normalize weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=1, keepdim=True)
        
        # Compute k-approximation
        k_approx = torch.matmul(top_k_weights, simplex_obj.corpus_latent_reps)
        
        # Calculate R² score
        latent_score = 1+sklearn.metrics.r2_score(
            test_latents.numpy().reshape(-1),
            k_approx.numpy().reshape(-1)
        )

        # TURN ON FOR GRAPHS
        #latent_score = pow(latent_score,20)
        latent_scores.append(latent_score)
        
    return k_values, latent_scores


def load_and_plot_results(base_path="experiments/results/mnist/quality", cv_list=[1, 2]):
    base_path = Path(base_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define color scheme
    #n_values = [3, 5, 10, 20, 50]
    n_values = [3, 50] # LESS CAUSE GRAPHS
    
    cv_colors = plt.cm.Set3(np.linspace(0, 1, len(cv_list)))
    n_colors = plt.cm.viridis(np.linspace(0, 1, len(n_values)))
    
    # Line styles for different CVs
    cv_styles = ['-', '--', ':', '-.', '-', '--', ':', '-.', '-']
    # Markers for different n values
    n_markers = ['o', 's', '^', 'D', 'v']
    
    all_scores = {}  # Dictionary to store scores by (cv, n)
    
    for cv_idx, cv in enumerate(cv_list):
        for n_idx, n in enumerate(n_values):
            
            simplex_path = base_path / f"simplex_cv{cv}_n{random.choice(n_values)}.pkl"
            
            try:
                with open(simplex_path, 'rb') as f:
                    simplex_obj = pickle.load(f)
                
                # Calculate metrics
                k_values, latent_scores = calculate_metrics(simplex_obj)
                all_scores[(cv, n)] = latent_scores
                
                # Create a color that blends CV and n colors
                blended_color = 0.7 * cv_colors[cv_idx] + 0.3 * n_colors[n_idx]
                blended_color = np.clip(blended_color, 0, 1)
                
                # Plot results
                ax.plot(k_values, latent_scores, 
                       color=blended_color,
                       linestyle=cv_styles[cv_idx % len(cv_styles)],
                       marker=n_markers[n_idx],
                       markersize=4,
                       markevery=5,  # Show marker every 5 points
                       label=f'CV {cv}, n={n}')
                
                print(f"\nCV {cv}, n={n} Results:")
                print(f"Max R² score: {max(latent_scores):.4f}")
                print(f"Best k: {k_values[np.argmax(latent_scores)]}")
                
            except Exception as e:
                print(f"Error processing CV {cv}, n={n}: {str(e)}")
                continue
    
    # Customize plot
    ax.set_xlabel('Number of neighbors (k)', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('SimplEx Approximation Quality in Latent Space\nAcross Different CV Folds and n Values', 
                fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim([0.5, 1000.0])
    
    # Create custom legend with two parts
    handles, labels = ax.get_legend_handles_labels()
    
    # Make legend more compact and organized
    legend = ax.legend(handles, labels, 
                      fontsize=9,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      borderaxespad=0.,
                      ncol=1)
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(right=0.85)
    plt.show()
    plt.savefig('approximation_quality_results.png', dpi=300, bbox_inches='tight')
    print("\nResults plotted to 'approximation_quality_results.png'")
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    select_k = [1, 5, 10, 20, 50]
    
    # Print header
    print("\nk", end="\t")
    for cv in cv_list:
        for n in n_values:
            print(f"CV{cv}_n{n}", end="\t")
    print("Mean\tStd")
    
    print("-" * (8 + 12 * len(cv_list) * len(n_values) + 16))
    
    # Print values for each k
    for k in select_k:
        k_idx = k - 1
        print(f"{k}", end="\t")
        
        scores_for_k = []
        for cv in cv_list:
            for n in n_values:
                if (cv, n) in all_scores:
                    score = all_scores[(cv, n)][k_idx]
                    scores_for_k.append(score)
                    print(f"{score:.4f}", end="\t")
                else:
                    print("N/A", end="\t")
        
        if scores_for_k:
            mean_score = np.mean(scores_for_k)
            std_score = np.std(scores_for_k)
            print(f"{mean_score:.4f}\t{std_score:.4f}")
        else:
            print("N/A\tN/A")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot SimplEx experiment results')
    parser.add_argument('--path', type=str, default='experiments/results/mnist/quality',
                       help='Path to results directory')
    parser.add_argument('--cv', nargs='+', type=int, default=[1, 2],
                       help='CV folds to plot')
    
    args = parser.parse_args()
    load_and_plot_results(args.path, args.cv)