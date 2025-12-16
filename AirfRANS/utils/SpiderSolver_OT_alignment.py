import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import ot
import scipy.io as io  

def compute_template(train_dataset):
    """Compute template point cloud"""
    template = np.zeros([1025, 2])
    point_total_num = np.zeros([1025]) + 1.0
    
    for i in range(len(train_dataset)):
        surf = train_dataset[i]['surf']
        point = train_dataset[i]['x'][surf, 0:2].numpy()
        index = train_dataset[i]['index']

        if len(index) == 1:
            template += point
        if len(index) != 1:
            cur = np.zeros([1025, 2])
            point_num = np.zeros([1025])
            cur[index] = point
            template += cur
            point_num[index] = 1.0
            point_total_num += point_num
  
    template = template / point_total_num.reshape(1025, 1)
    return template


def perform_spectral_clustering(template, n_clusters):
    """Perform spectral clustering and return results"""
    model_Clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    labels_SpectralClustering = model_Clustering.fit_predict(template)
    return labels_SpectralClustering


def save_template_and_plot(template, labels_SpectralClustering, path, current_file_path):
    """Save template data and plot clustering results"""
    io.savemat(path + 'template.mat', {'template': template, 'labels_SpectralClustering': labels_SpectralClustering})
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(template[:, 0], template[:, 1], c=labels_SpectralClustering)
    plt.colorbar(sc, ax=ax, label='Height (Z-axis)')
    plt.savefig(path + "template.png", dpi=800, bbox_inches='tight')
    plt.close()


def mode(row):
    """Calculate mode (most frequent value)"""
    unique_elements, counts = np.unique(row, return_counts=True)
    return unique_elements[np.argmax(counts)]


def assign_labels_to_pointcloud(point, template, labels_SpectralClustering):
    """Assign labels to point cloud"""
    a = np.ones([template.shape[0]]) / template.shape[0]
    b = np.ones([point.shape[0]]) / point.shape[0]
    M = ot.dist(template, point, metric='euclidean')
    ot_matrix = ot.sinkhorn(a, b, M, reg=0.8, numItermax=25)
    index = np.argmax(ot_matrix, axis=0)
    current_labels = labels_SpectralClustering[index]
    
    
    # Assign spectral clustering labels to point cloud with neighbor voting
    dist_matrix = cdist(point, point, metric='euclidean')
    nearest_neighbors_indices = np.argsort(dist_matrix, axis=1)[:, 1:10+1]
    neighbor_labels = current_labels[nearest_neighbors_indices]
    most_frequent = np.apply_along_axis(mode, axis=1, arr=neighbor_labels)
    
    return most_frequent, index


def process_dataset(dataset, template, labels_SpectralClustering, path):
    """Process all point clouds in dataset"""
    for i in range(len(dataset)):
        surf = dataset[i]['surf']
        point = dataset[i]['x'][surf, 0:2].numpy()
        most_frequent, index = assign_labels_to_pointcloud(point, template, labels_SpectralClustering)
        
        dataset[i]['labels_SpectralClustering'] = most_frequent
        dataset[i]['index'] = index


        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # sc = ax.scatter(point[:, 0], point[:, 1], c=most_frequent)
        # plt.colorbar(sc, ax=ax, label='Height (Z-axis)')
        # plt.savefig(path + str(i) + ".png", dpi=800, bbox_inches='tight')
        # plt.close()
