#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hdbscan
from sklearn.datasets import make_blobs

import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import pi, cos, sin, atan2, sqrt



def test1():
    data, _ = make_blobs(1000)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = clusterer.fit_predict(data)

def test2():
    data, _ = make_blobs(1000)
    clusterer = hdbscan.RobustSingleLinkage(cut=0.125, k=7)
    cluster_labels = clusterer.fit_predict(data)
    hierarchy = clusterer.cluster_hierarchy_
    alt_labels = hierarchy.get_clusters(0.100, 5)
    hierarchy.plot()


def get_centroid(cluster):
    x = y = z = 0
    coord_num = len(cluster)
    for coord in cluster:
        lat = coord[0] * pi / 180
        lon = coord[1] * pi / 180

        a = cos(lat) * cos(lon)
        b = cos(lat) * sin(lon)
        c = sin(lat)

        x += a
        y += b
        z += c
    x /= coord_num
    y /= coord_num
    z /= coord_num
    lon = atan2(y, x)
    hyp = sqrt(x * x + y * y)
    lat = atan2(z, hyp)
    return [lat * 180 / pi, lon * 180 / pi]

def test3():
    import numpy as np

    from hdbscan import HDBSCAN
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler

    import time

    def make_var_density_blobs(n_samples=750, centers=[[0, 0]], cluster_std=[0.5], random_state=0):
        samples_per_blob = n_samples // len(centers)
        blobs = [make_blobs(n_samples=samples_per_blob, centers=[c], cluster_std=cluster_std[i])[0]
                 for i, c in enumerate(centers)]
        labels = [i * np.ones(samples_per_blob) for i in range(len(centers))]
        return np.vstack(blobs), np.hstack(labels)

    ##############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    densities = [0.2, 0.35, 0.5]
    X, labels_true = make_var_density_blobs(n_samples=750, centers=centers, cluster_std=densities,
                                            random_state=0)

    X = StandardScaler().fit_transform(X)

    ##############################################################################
    # Compute DBSCAN
    hdb_t1 = time.time()
    hdb = HDBSCAN(min_cluster_size=10).fit(X)
    hdb_labels = hdb.labels_
    hdb_elapsed_time = time.time() - hdb_t1

    db_t1 = time.time()
    db = DBSCAN(eps=0.1).fit(X)
    db_labels = db.labels_
    db_elapsed_time = time.time() - db_t1

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

    print('\n\n++ HDBSCAN Results')
    print('Estimated number of clusters: %d' % n_clusters_hdb_)
    print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
    print('Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, hdb_labels))
    print('Completeness: %0.3f' % metrics.completeness_score(labels_true, hdb_labels))
    print('V-measure: %0.3f' % metrics.v_measure_score(labels_true, hdb_labels))
    print('Adjusted Rand Index: %0.3f'
          % metrics.adjusted_rand_score(labels_true, hdb_labels))
    print('Adjusted Mutual Information: %0.3f'
          % metrics.adjusted_mutual_info_score(labels_true, hdb_labels))
    print('Silhouette Coefficient: %0.3f'
          % metrics.silhouette_score(X, hdb_labels))

    n_clusters_db_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    print('\n\n++ DBSCAN Results')
    print('Estimated number of clusters: %d' % n_clusters_db_)
    print('Elapsed time to cluster: %.4f s' % db_elapsed_time)
    print('Homogeneity: %0.3f' % metrics.homogeneity_score(labels_true, db_labels))
    print('Completeness: %0.3f' % metrics.completeness_score(labels_true, db_labels))
    print('V-measure: %0.3f' % metrics.v_measure_score(labels_true, db_labels))
    print('Adjusted Rand Index: %0.3f'
          % metrics.adjusted_rand_score(labels_true, db_labels))
    print('Adjusted Mutual Information: %0.3f'
          % metrics.adjusted_mutual_info_score(labels_true, db_labels))
    if n_clusters_db_ > 1:
        print('Silhouette Coefficient: %0.3f'
              % metrics.silhouette_score(X, db_labels))
    else:
        print('Silhouette Coefficient: NaN (too few clusters)')

    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    hdb_unique_labels = set(hdb_labels)
    db_unique_labels = set(db_labels)
    hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
    db_colors = plt.cm.Spectral(np.linspace(0, 1, len(db_unique_labels)))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    hdb_axis = fig.add_subplot('121')
    db_axis = fig.add_subplot('122')
    for k, col in zip(hdb_unique_labels, hdb_colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        hdb_axis.plot(X[hdb_labels == k, 0], X[hdb_labels == k, 1], 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=6)
    for k, col in zip(db_unique_labels, db_colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        db_axis.plot(X[db_labels == k, 0], X[db_labels == k, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

    hdb_axis.set_title('HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
    db_axis.set_title('DBSCAN\nEstimated number of clusters: %d' % n_clusters_db_)
    plt.show()

def test4():
    import time

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import cluster, datasets
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler

    import hdbscan

    np.random.seed(0)
    plt.style.use('fivethirtyeight')

    def make_var_density_blobs(n_samples=750, centers=[[0, 0]], cluster_std=[0.5], random_state=0):
        samples_per_blob = n_samples // len(centers)
        blobs = [datasets.make_blobs(n_samples=samples_per_blob, centers=[c], cluster_std=cluster_std[i])[0]
                 for i, c in enumerate(centers)]
        labels = [i * np.ones(samples_per_blob) for i in range(len(centers))]
        return np.vstack(blobs), np.hstack(labels)

    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.08)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.10)
    blobs = datasets.make_blobs(n_samples=n_samples - 200, random_state=8)
    noisy_blobs = np.vstack((blobs[0], 25.0 * np.random.rand(200, 2) - [10.0, 10.0])), np.hstack(
        (blobs[1], -1 * np.ones(200)))
    varying_blobs = make_var_density_blobs(n_samples,
                                           centers=[[1, 1],
                                                    [-1, -1],
                                                    [1, -1]],
                                           cluster_std=[0.2, 0.35, 0.5])
    no_structure = np.random.rand(n_samples, 2), None

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)

    clustering_names = [
        'MiniBatchKMeans', 'AffinityPropagation',
        'SpectralClustering', 'AgglomerativeClustering',
        'DBSCAN', 'HDBSCAN']

    plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    datasets = [noisy_circles, noisy_moons, noisy_blobs, varying_blobs, no_structure]
    for i_dataset, dataset in enumerate(datasets):
        X, y = dataset
        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # create clustering estimators
        two_means = cluster.MiniBatchKMeans(n_clusters=2)
        spectral = cluster.SpectralClustering(n_clusters=2,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=.2)
        affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                           preference=-200)

        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock", n_clusters=2,
            connectivity=connectivity)

        hdbscanner = hdbscan.HDBSCAN()
        clustering_algorithms = [
            two_means, affinity_propagation, spectral, average_linkage,
            dbscan, hdbscanner]

        for name, algorithm in zip(clustering_names, clustering_algorithms):
            # predict cluster memberships
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            # plot
            plt.subplot(5, len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

            if hasattr(algorithm, 'cluster_centers_'):
                centers = algorithm.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()

def main():
    # test2()
    # test3()
    test4()


if __name__ == '__main__':
    main()