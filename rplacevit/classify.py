import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
from .dataset import RPlaceDataset

def classify_and_visualize_users(dataset, dimensions=2, n_clusters=100, output_file='user_classification.png'):
    """
    Classify users based on their features and visualize them in 2D or 3D.
    
    Args:
        dataset (RPlaceDataset): The dataset containing user features
        dimensions (int): Number of dimensions for visualization (2 or 3)
        n_clusters (int): Number of clusters for KMeans
        output_file (str): Name of the output PNG file
    """
    if not hasattr(dataset, 'user_features'):
        dataset.compute_users_features()
    
    user_ids = list(dataset.user_features.keys())
    features = np.array(list(dataset.user_features.values()))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features)
    labels = kmeans.labels_
    
    pca = PCA(n_components=dimensions)
    reduced_features = pca.fit_transform(features)
    
    fig = plt.figure(figsize=(16, 14))
    
    cmap = plt.get_cmap('viridis')
    
    if dimensions == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                             c=labels, cmap=cmap, alpha=0.6, s=10)
        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2],
                             c=labels, cmap=cmap, alpha=0.6, s=10)
        ax.set_xlabel('First Principal Component', fontsize=12)
        ax.set_ylabel('Second Principal Component', fontsize=12)
        ax.set_zlabel('Third Principal Component', fontsize=12)
    else:
        raise ValueError("Dimensions must be 2 or 3")
    
    plt.title(f'{dimensions}D Visualization of User Classification\n({n_clusters} clusters)', fontsize=16)
    
    cbar = plt.colorbar(scatter, ax=ax, aspect=30)
    cbar.set_label('Cluster', fontsize=12)
    
    cbar.ax.tick_params(labelsize=10)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {output_file}")
    
    with open('user_classification_data.pkl', 'wb') as f:
        pickle.dump({
            'user_ids': user_ids, 
            'reduced_features': reduced_features,
            'cluster_labels': labels
        }, f)
    print("User classification data saved as user_classification_data.pkl")

if __name__ == "__main__":
    dataset = RPlaceDataset()
    classify_and_visualize_users(dataset, dimensions=2, n_clusters=100, output_file='user_classification_2d.png')
    classify_and_visualize_users(dataset, dimensions=3, n_clusters=100, output_file='user_classification_3d.png')