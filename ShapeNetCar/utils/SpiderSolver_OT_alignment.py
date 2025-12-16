import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import scipy.io as io
import ot


def compute_template(train_data):
    """Compute template point cloud from training data"""
    template = np.zeros([3586, 3])

    for i in range(len(train_data)):
        surf = train_data[i]['surf']
        point = train_data[i]['x'][surf, 0:3]
        point = np.concatenate((point[0:16,], point[112:]), axis=0)
        template += point

    template = template / len(train_data)
    return template


def perform_spectral_clustering(template, n_clusters):
    """Perform spectral clustering on template points"""
    model_Clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    labels_SpectralClustering = model_Clustering.fit_predict(template)
    return labels_SpectralClustering


def save_template_and_plot(template, labels_SpectralClustering, path, coef_norm):
    """Save template data and create 3D visualization"""
    io.savemat(path + 'template.mat', 
            {'template': template, 'labels_SpectralClustering': labels_SpectralClustering})
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points_mean = coef_norm[0][0:3].reshape(1, 3)
    points_std = coef_norm[1][0:3].reshape(1, 3)
    template_normalized = (template * points_std) + points_mean
    
    # Create scatter plot with elevation coloring
    scfig = ax.scatter(template_normalized[:, 0], template_normalized[:, 2], template_normalized[:, 1], 
                      c=labels_SpectralClustering, cmap='viridis', marker='o')
    
    # Add colorbar and labels
    plt.colorbar(scfig, ax=ax, label='Height (Z-axis)')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-2.2, 2.2)
    
    plt.savefig(path + "template.png", dpi=300, pad_inches=0)  
    plt.close()


def mode(row):
    """Calculate the mode (most frequent value) in a row"""
    unique_elements, counts = np.unique(row, return_counts=True)
    return unique_elements[np.argmax(counts)]


def assign_labels_to_pointcloud(point, template, labels_SpectralClustering):
    """Assign labels to point cloud"""

    a = np.ones([template.shape[0]]) / template.shape[0]
    b = np.ones([point.shape[0]]) / point.shape[0]
    M = ot.dist(template, point, metric='euclidean')
    ot_matrix = ot.sinkhorn(a, b, M, reg=0.8, numItermax=50)
    index = np.argmax(ot_matrix, axis = 0)
    current_labels = labels_SpectralClustering[index]
    
    # Assign spectral clustering labels to point cloud with neighbor voting
    dist_matrix = cdist(point, point, metric='euclidean')
    nearest_neighbors_indices = np.argsort(dist_matrix, axis=1)[:, 1:10+1]
    neighbor_labels = current_labels[nearest_neighbors_indices]
    most_frequent = np.apply_along_axis(mode, axis=1, arr=neighbor_labels)
    
    return most_frequent


def process_dataset(dataset, template, labels_SpectralClustering, coef_norm, path):
    """Process dataset and assign spectral clustering labels to each point cloud"""
    for i in range(len(dataset)):
        surf = dataset[i]['surf']
        point = dataset[i]['x'][surf, 0:3]
        point = np.concatenate((point[0:16,], point[112:]), axis=0)

        most_frequent = assign_labels_to_pointcloud(point, template, labels_SpectralClustering)
        dataset[i]['labels_SpectralClustering'] = most_frequent
        
        # # Create 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # points_mean = coef_norm[0][0:3].reshape(1, 3)
        # points_std = coef_norm[1][0:3].reshape(1, 3)
        # point_normalized = (point * points_std) + points_mean
        
        # # Create scatter plot with elevation coloring
        # scfig = ax.scatter(point_normalized[:, 0], point_normalized[:, 2], point_normalized[:, 1], 
        #                 c=most_frequent, cmap='viridis', marker='o')
        
        # # Add colorbar and labels
        # plt.colorbar(scfig, ax=ax, label='Height (Z-axis)')
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # ax.set_xlim(-2.2, 2.2)
        # ax.set_ylim(-2.2, 2.2)
        # ax.set_zlim(-2.2, 2.2)
        
        # plt.savefig(path + str(i) + ".png", dpi=300, pad_inches=0)  
        # plt.close()
        
        
        
        

