import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fungsi untuk menghitung jarak Euclidean antara dua vektor
def hitung_jarak(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# Fungsi utama untuk segmentasi dengan algoritma KNN
def knn_segmentasi_gambar(citra, k, max_iterasi=100):
    # Mengubah citra menjadi array satu dimensi
    citra_flat = citra.reshape((-1, 3))

    # Inisialisasi titik data secara acak ke k cluster
    titik_data = np.random.randint(citra_flat.shape[0], size=k)
    centroid = citra_flat[titik_data]

    iterasi = 0
    while iterasi < max_iterasi:
        # Menghitung jarak titik data dari pusat masing-masing cluster
        jarak = np.zeros((citra_flat.shape[0], k))
        for i in range(k):
            jarak[:, i] = np.linalg.norm(citra_flat - centroid[i], axis=1)

        # Mengelompokkan setiap titik data ke klaster terdekat
        klaster_terdekat = np.argmin(jarak, axis=1)

        # Menyimpan centroid lama untuk memeriksa konvergensi
        centroid_lama = centroid.copy()

        # Memperbarui centroid klaster
        for i in range(k):
            centroid[i] = np.mean(citra_flat[klaster_terdekat == i], axis=0)

        # Memeriksa konvergensi
        if np.array_equal(centroid, centroid_lama):
            break

        iterasi += 1

    # Mengganti warna setiap piksel dengan warna dari centroid klaster terdekat
    citra_segmented = centroid[klaster_terdekat].astype(np.uint8)

    # Mengembalikan citra hasil segmentasi dalam bentuk asli (dengan bentuk dan ukuran yang sama)
    citra_segmented = citra_segmented.reshape(citra.shape)

    return citra_segmented

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title('UAS Pengolahan Citra Digital')
    st.title('Nama: Selma Ohoira')
    st.title('NIM : 312210737')
    st.title('Kelas: TI22C6')
    st.title('Segmentasi Gambar dengan Algoritma KNN')
    
    uploaded_file = st.file_uploader("Upload sebuah gambar", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.image(img, caption='Gambar Asli', use_column_width=True)
        
        k = st.slider("Pilih jumlah cluster (k)", 2, 10, 3)
        max_iterasi = st.slider("Jumlah iterasi maksimum", 10, 200, 100)
        
        if st.button("Segmentasi"):
            # Konversi warna citra dari BGR ke RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Segmentasi citra dengan KNN
            img_segmented = knn_segmentasi_gambar(img_rgb, k=k, max_iterasi=max_iterasi)
            
            # Menampilkan hasil segmentasi
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Citra Asli')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(img_segmented)
            plt.title('Hasil Segmentasi dengan KNN')
            plt.axis('off')

            # Create a Matplotlib figure and axis
            fig, ax = plt.subplots()

            # Example plot (replace with your own plotting code)
            ax.plot([1, 2, 3], [4, 5, 6])
            ax.set_title('Example Plot')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            st.pyplot()
            
if __name__ == '__main__':
    main()