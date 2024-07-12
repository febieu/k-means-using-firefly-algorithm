import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# max_iter = 60
# alpha = 0.1 
# beta_0 = 0.1 
# gamma = 0.01 
# xmin = -5.0
# xmax = 5.0

def hitung_objektif(populasi, n_cluster):
    km = KMeans(n_clusters=n_cluster, init='random', max_iter=60, n_init=1, random_state=None)
    labels = km.fit_predict(populasi)
    centroids = km.cluster_centers_
    sil_score = silhouette_score(populasi, labels, metric='euclidean')
    return centroids, sil_score, labels

def get_best_obj_kmeans(index, labels, centroid, data):
    if 0 <= index < len(labels):
        plot_clusters(data, labels[index], centroid[index])


def plot_clusters(data, labels, centroid=None):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=80)
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, s=50, cmap='viridis')
    
    # Buat centroid 
    if centroid is not None:
        ax.scatter(centroid[:, 0], centroid[:, 1], c='red', s=200, alpha=0.75, label='Centroid')
    
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.set_title('KMeans Clustering')
    ax.legend()
    st.pyplot(fig)

#Firefly code
def initialize_fireflies(population):
    lb = np.min(population)
    ub = np.max(population)
    x = np.random.uniform(low=lb, high=ub, size=(population.shape[0] , population.shape[1] ))
    return x

def hitung_intensitas (populasi, obj):
    intensitas = np.full((populasi.shape[0], populasi.shape[1]), obj)
    return intensitas

def hitung_jarak(firefly_i, firefly_j):
    return np.linalg.norm(firefly_i - firefly_j)

def hitung_daya_tarik(distance_ij, beta_0, gamma):
    return beta_0 * np.exp(-gamma * distance_ij**2)

def move_fireflies (pop, max_int, cluster, int_pop, centroid, objektif, a, best_centroid, best_labels, alpha,beta_0, gamma, ):
    if (int_pop[0] < max_int[0]).all():
        # print("int nya lebih kecil")
        for i in range(pop.shape[0]):  
            for j in range(pop.shape[1]): 
                distance_ij = hitung_jarak(pop[i], a[j])
                attractiveness_ij = hitung_daya_tarik(distance_ij, beta_0, gamma)
                alpha_vector = np.repeat(alpha, pop.shape[1])
                random_value = np.random.rand() - 0.5
                pop[i][j] += attractiveness_ij * (a[i][j] - pop[i][j]) + alpha_vector[j] * random_value
            # st.subheader("Pop:")
            # st.write(pop)                
        centroid, objektif, labels = hitung_objektif(pop, cluster)
        int_pop = hitung_intensitas(pop, objektif)
        # st.subheader("Int pop:")
        # st.write(int_pop)
        
        # Update max_int, best_position, dan best_centroid 
        if (int_pop[0] > max_int[0]).all():
            max_int = int_pop.copy()
            a = pop.copy()
            best_centroid = centroid.copy()
            # best_cluster = 2
            best_labels = labels.copy()
        # print(max_int[0,0])

    elif (int_pop[0] >= max_int[0]).all():
        # print("int nya lebih besar")        
        for i in range(pop.shape[0]):
            for j in range(pop.shape[1]):
                pop[i, j] = pop[i, j] + alpha * (np.random.rand(*pop[i, j].shape) - 0.5)
            # st.subheader("Pop:")
            # st.write(pop)
        centroid, objektif, labels = hitung_objektif(pop, cluster)
        int_pop = hitung_intensitas(pop, objektif)
        # st.subheader("Int pop:")
        # st.write(int_pop)
        # Update max_int, best_position, dan best_centroid
        if (int_pop[0] > max_int[0]).all():
            max_int = int_pop.copy()
            a = pop.copy()
            best_centroid = centroid.copy()
            # best_cluster = 2
            best_labels = labels.copy()
        # print(max_int[0,0])
    return pop, max_int, best_centroid, best_labels

def plot_clusters_fireflies(data, labels, centroid=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    
    # Plotting the standard cluster centers (if available)
    if centroid is not None:
        ax.scatter(centroid[:, 0], centroid[:, 1], c='red', s=200, alpha=0.75, label='Centroid')
    
    ax.set_xlabel("fixed acidity") 
    ax.set_ylabel("volatile acidity")
    ax.set_title('KMeans Clustering with Firefly')
    ax.legend()
    st.pyplot(fig)
