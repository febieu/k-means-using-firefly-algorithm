import numpy as np
import pandas as pd
import streamlit as st
from function import hitung_objektif, get_best_obj_kmeans, initialize_fireflies, move_fireflies, plot_clusters_fireflies

max_iter = 60
alpha = 0.1
beta_0 = 0.1
gamma = 0.01

st.set_page_config(
    page_title="Clustering K-Means dan Firefly",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Perhitungan Clustering K-Means dan Algoritma Firefly")

file = st.file_uploader("Upload file CSV atau Excel", type=['csv', 'xlsx'])

if file is not None:
    if file.name.endswith('csv'):
        data = pd.read_csv(file)
    elif file.name.endswith(('xls', 'xlsx')):
        data = pd.read_excel(file)
    else:
        st.error("Format file tidak didukung. Silakan unggah file CSV atau Excel.")

if 'data' in locals() or 'data' in globals():
    #pembersihan data
    data_cleaning = data.dropna() #menghapus nilai = 0
    data_dup = data_cleaning.drop_duplicates() # nilai yang duplicates

    # Menghapus outlier (titik data yg berbeda dari data umumnya) 
    Q1 = data_dup.quantile(0.25)
    Q3 = data_dup.quantile(0.75)
    IQR = Q3 - Q1
    data_c = data_dup[~((data_dup < (Q1 - 1.5 * IQR)) | (data_dup > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Menyimpan data yang telah dibersihkan ke file baru
    data_c.to_csv("data_bersih.csv", index=False)

    #normalisasi
    min_val = np.min(data_c, axis=0)
    max_val = np.max(data_c, axis=0)
    normalized_data = (data_c - min_val) / (max_val - min_val)

    #membuat 2 column di website 
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data sebelum normalisasi: ", data)
    with col2:
        st.write("Data setelah normalisasi:", normalized_data)

    if st.button('Hitung'):
       # --------- K - M E A N S ------------ C O D E -----------
        st.subheader("Perhitungan Clustering K-Means:")
        results_kmeans = []
        for i in range(2, 9):
            cen_kmeans, obj_kmeans, labels_kmeans = hitung_objektif(normalized_data, i)
            results_kmeans.append((cen_kmeans, obj_kmeans, labels_kmeans))

        cen_kmeans_sc = [result[0] for result in results_kmeans]
        obj_kmeans_sc = [result[1] for result in results_kmeans]
        labels_kmeans_sc = [result[2] for result in results_kmeans]

        array_max_obj_kmeans = np.array([obj_kmeans_sc])
        index_array_obj_kmeans = np.argmax(array_max_obj_kmeans)

        st.write("Nilai Max Silhouette Score: ",array_max_obj_kmeans[0][index_array_obj_kmeans])

        col1, col2 = st.columns(2)
        with col1:
            st.write("Titik Populasi dan labelnya: ")
            df_kmeans = pd.DataFrame(normalized_data)
            df_kmeans['Labels'] = labels_kmeans_sc[index_array_obj_kmeans]
            st.write(df_kmeans)
        with col2:
            st.write("Titik Centroid: ")
            st.write(cen_kmeans_sc[index_array_obj_kmeans])

        get_best_obj_kmeans(index_array_obj_kmeans, labels_kmeans_sc, cen_kmeans_sc, normalized_data)

        # --------- F I R E F L Y ------------ C O D E -----------
        st.header("Perhitungan Clustering K-Means dengan Firefly:")
        data_pop_1 = [initialize_fireflies(normalized_data) for _ in range(8)]
        # st.write(data_pop_1[0]) --> data 1 populasi 1
        data_pop_2 = [initialize_fireflies(normalized_data) for _ in range(8)]

        #SIMPEL CODE
        results_firefly_1 = []
        for i, pop in enumerate(data_pop_1, start=2):
            cen, obj, labels = hitung_objektif(pop, i)
            results_firefly_1.append((cen, obj, labels))
        results_firefly_2 = []
        for i, pop in enumerate(data_pop_2, start=2):
            cen_02, obj_02, labels_02 = hitung_objektif(pop, i)
            results_firefly_2.append((cen_02, obj_02, labels_02))

        cen_fireflies_1 = [result[0] for result in results_firefly_1]
        cen_fireflies_2 = [result[0] for result in results_firefly_2]
        obj_fireflies_1 = [result[1] for result in results_firefly_1]
        obj_fireflies_2 = [result[1] for result in results_firefly_2]
        labels_fireflies_1 = [result[2] for result in results_firefly_1]
        labels_fireflies_2 = [result[2] for result in results_firefly_2]

        #SIMPEL CODE
        int_arrs_1 = [np.full((pop.shape[0], pop.shape[1]), obj) for pop, obj in zip(data_pop_1, obj_fireflies_1)]
        int_arrs_2 = [np.full((pop.shape[0], pop.shape[1]), obj) for pop, obj in zip(data_pop_2, obj_fireflies_2)]

        int_values_1 = [int_arr[0, 0] for int_arr in int_arrs_1]
        int_values_2 = [int_arr[0, 0] for int_arr in int_arrs_2]
        
        #membuat array yg isinya satu intensitas dri masing2 data tiap populasi
        arrs_int_1 = np.array(int_values_1)
        arrs_int_2 = np.array(int_values_2)

        #membuat salinan data_pop_1 dan data_pop_2
        data_c1 = [pop.copy() for pop in data_pop_1]
        data_c2 = [pop.copy() for pop in data_pop_2]

        #salinan centroid
        cen_c1 = [cen.copy() for cen in cen_fireflies_1]
        cen_c2 = [cen.copy() for cen in cen_fireflies_2]

        #dan labels
        lab_c1 = [lab.copy() for lab in labels_fireflies_1]
        lab_c2 = [lab.copy() for lab in labels_fireflies_2]

        #variabel sbg 
        value =  float('-inf')
        #Definisikan variabel diluar kondisi:
        int_max = int_arrs_1.copy()
        best_pop = data_pop_1[0].copy()
        best_centroid = cen_fireflies_1[0].copy()
        best_labels = labels_fireflies_1[0].copy()

        max_arrs_1, idx_arrs_1 = max((v,i) for i, v in enumerate(arrs_int_1))
        max_arrs_2, idx_arrs_2 = max((v,i) for i, v in enumerate(arrs_int_2))

        if max_arrs_1 > max_arrs_2:
            int_max = int_arrs_1[idx_arrs_1].copy()
            best_pop = data_pop_1[idx_arrs_1].copy()
            best_centroid = cen_fireflies_1[idx_arrs_1].copy()
            best_labels = labels_fireflies_1[idx_arrs_1].copy()

        elif max_arrs_2 > max_arrs_1:
            int_max = int_arrs_2[idx_arrs_2].copy()
            best_pop = data_pop_2[idx_arrs_2].copy()
            best_centroid = cen_fireflies_2[idx_arrs_2].copy()
            best_labels = labels_fireflies_2[idx_arrs_2].copy()

        #SIMPEL CODE FOR MOVE POPULATION
        fireflies_values_1 = []
        fireflies_values_2 = []
        for i in range (max_iter):
            for cluster in range(8):
                # st.write(cluster)
                pop_1, int_1, cen_1, lab_1 = move_fireflies(data_c1[cluster], int_max, cluster+2, int_arrs_1[cluster], cen_c1[cluster], obj_fireflies_1[cluster], best_pop, best_centroid, best_labels, alpha, beta_0, gamma)
                fireflies_values_1.append((pop_1, int_1, cen_1, lab_1))
                pop_2, int_2, cen_2, lab_2 = move_fireflies(data_c2[cluster], int_max, cluster+2, int_arrs_2[cluster], cen_c2[cluster], obj_fireflies_2[cluster], best_pop, best_centroid, best_labels, alpha, beta_0, gamma)
                fireflies_values_2.append((pop_2, int_2, cen_2, lab_2))

            # total_1_c2 = fireflies_values_1[0]
            # values_1_c2 = total_1_c2[1]

            # total_1_c3 = fireflies_values_1[1]
            # values_1_c3 = total_1_c3[1]

            des_val_1 = []
            des_val_2 = []
            for i in range (8):
                a = fireflies_values_1[i][1][0,0]
                b = fireflies_values_2[i][1][0,0]
                des_val_1.append(a)
                des_val_2.append(b)

            max_des_1, idx_des_1 = max((v,i) for i, v in enumerate(des_val_1))
            max_des_2, idx_des_2 = max((v,i) for i, v in enumerate(des_val_2))

            if(max_des_1 > max_des_2):
                # st.write("masuk ke pop 1:")
                var_int = fireflies_values_1[idx_des_1][1].copy()
                var_pop = fireflies_values_1[idx_des_1][0].copy()
                var_cent = fireflies_values_1[idx_des_1][2].copy()
                var_lab = fireflies_values_1[idx_des_1][3].copy()
            elif(max_des_2 > max_des_1):
                # st.write("masuk ke pop 2:")
                var_int = fireflies_values_2[idx_des_2][1].copy()
                var_pop = fireflies_values_2[idx_des_2][0].copy()
                var_cent = fireflies_values_2[idx_des_2][2].copy()
                var_lab = fireflies_values_2[idx_des_2][3].copy()
            elif(max_des_1 == max_des_2):
                # st.write("sama!")
                var_int = int_max.copy()
                var_pop = best_pop.copy()
                var_cent = best_centroid.copy()
                var_lab = best_labels.copy()

        st.write("G-best dari Algoritma Firefly: ", var_int[0,0])
        col1, col2 = st.columns(2)
        with col1:
            st.write("Titik data dan Labelnya:")
            df_fireflies = pd.DataFrame(var_pop)
            df_fireflies['Labels'] = var_lab
            st.write(df_fireflies)    
        with col2:
            st.write("Titik centroid: ")
            st.write(var_cent)

        plot_clusters_fireflies(var_pop, var_lab, var_cent)
