from importlib import reload
import scipy.io as sio
import plotly.io as pio
pio.renderers.default='browser'
import pandas as pd
import numpy as np
import open3d as o3d
import numpy as np
import torch

def SpiderSolver_BloodFlow_DataProcess(points, n_clusters, onion_num, sava_path):
    points = np.array(points)
    
    # Create open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Surface reconstruction: Extract surface using Alpha Shape algorithm
    alpha = 0.005  # Alpha value needs to be adjusted according to point cloud density
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # o3d.visualization.draw_geometries([mesh])
    
    # Uniform sampling of the surface
    sampled_points = mesh.sample_points_uniformly(number_of_points=10000)
    # o3d.visualization.draw_geometries([sampled_points])
    sampled_points = np.asarray(sampled_points.points)
    # Visualize sampled points
    
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, sampled_points, metric='euclidean') 
    sdf = np.min(dist_matrix, axis = 1)
    
    sdf_sort = np.sort(sdf)
    sdf_sort_index = np.argsort(sdf)
    point_surface_dataset = points[np.sort(sdf_sort_index[0:456])]
    
    onion_index_0 = np.zeros([sdf.shape[0]])
    onion_index_0[sdf_sort_index[456: 456+200 ]] = 1.0
    onion_index_0 = torch.from_numpy(onion_index_0)
    
    from sklearn.cluster import SpectralClustering
    # Create Spectral Clustering model, select required number of clusters k
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    # Fit data and make predictions
    labels_SpectralClustering = model.fit_predict(point_surface_dataset)
    
    import matplotlib.pyplot as plt
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Elevation rendering: Use z-axis values as colors
    sc = ax.scatter(point_surface_dataset[:, 0], point_surface_dataset[:, 1], point_surface_dataset[:, 2], c=labels_SpectralClustering, cmap='viridis', marker='o')
    # Add color bar showing z-axis elevation mapping
    plt.colorbar(sc, ax=ax, label='Height (Z-axis)')
    plt.savefig(sava_path + 'clusters_' + str(n_clusters)  + ".png", dpi=300, pad_inches=0)  
    
    ## ## ## ## ## ## ## ## ## Onion Rings ## ## ## ## ## ## ## ## ## ##
    point_num_list_1 = np.ones(1)*456
    point_num_list_2 = np.ones(onion_num-1)* int(1200/onion_num)
    point_num_list_3 = [sdf.shape[0] - 456 - (onion_num-1)*int(1200/onion_num)]
    onion_slice_num = 1+onion_num
    point_num_list = np.concatenate([point_num_list_1, point_num_list_2, point_num_list_3], axis = 0).astype(int)
    
    onion_index = np.zeros([onion_slice_num, sdf.shape[0]])
    counter = 0 
    for i in range(onion_slice_num-1):
        zeros = np.zeros(sdf.shape[0])
        zeros[sdf_sort_index[counter: counter+point_num_list[i] ]] = 1.0
        onion_index[i,:] = zeros
        counter = counter + point_num_list[i]
    # print(counter)
    zeros = np.zeros(sdf.shape[0])
    zeros[sdf_sort_index[counter: ]] = 1.0
    onion_index[onion_slice_num-1,:] = zeros
    
    print(np.sum(onion_index))
    print(np.sum(onion_index, axis = 1))
    
    onion_index2 = np.argmax(onion_index, axis = 0)
    onion_index = onion_index[1:,:]
    
    surf_points = points[np.sort(sdf_sort_index[0:456])]
    velo_points = points[np.sort(sdf_sort_index[456:])]
    
    surf = np.zeros([1656])
    surf_index = sdf_sort_index[0:456]
    surf_index = np.sort(surf_index)
    surf[surf_index] = 1
    
    dist_matrix2 = cdist(surf_points, velo_points, metric='euclidean') 
    closest_indices_on_surface = np.argmin(dist_matrix2, axis = 0)
    velo_cluster_index = labels_SpectralClustering[closest_indices_on_surface]
    
    # Convert to one-hot encoding
    velo_cluster_index_one_hot = np.eye(n_clusters)[velo_cluster_index]  # Output shape (1200, 4)
    
    surf = torch.from_numpy(surf)
    onion_index = torch.from_numpy(onion_index)
    velo_cluster_index_one_hot = torch.from_numpy(velo_cluster_index_one_hot)
    velo_cluster_index_one_hot = velo_cluster_index_one_hot.permute(1,0)
    
    return surf, onion_index, onion_index_0, velo_cluster_index_one_hot, labels_SpectralClustering