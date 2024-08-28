import streamlit as st
import pandas as pd
import joblib

# Judul aplikasi
st.title("Prediksi Harga Rumah")
st.write("""
Aplikasi ini memprediksi harga rumah berdasarkan fitur-fitur seperti jumlah kamar, luas, tahun dibangun, dan lain-lain.
""")

# Memuat data
@st.cache_data
def load_data():
    data = pd.read_csv('house_prediction.csv')
    return data

data = load_data()

# Mendapatkan opsi unik untuk kolom alamat
def get_unique_options(df, column_name, filter_by=None, filter_value=None):
    if filter_by and filter_value:
        df = df[df[filter_by] == filter_value]
    return df[column_name].dropna().unique().tolist()

# Filter data berdasarkan kolom yang dipilih
def filter_data(df, **filters):
    filtered_data = df
    for column, value in filters.items():
        if value:
            filtered_data = filtered_data[filtered_data[column] == value]
    return filtered_data

def user_input_features():
    # Pilihan otomatis untuk alamat
    country_options = get_unique_options(data, 'country')
    country = st.sidebar.selectbox('Negara', options=country_options, index=0)

    # Filter data berdasarkan negara
    filtered_data = filter_data(data, country=country)

    city_options = get_unique_options(filtered_data, 'city')
    city = st.sidebar.selectbox('Kota', options=city_options, index=0)
    
    # Filter data berdasarkan kota
    filtered_data = filter_data(filtered_data, city=city)

    street_options = get_unique_options(filtered_data, 'street')
    street = st.sidebar.selectbox('Alamat Jalan', options=street_options, index=0)
    
    # Filter data berdasarkan jalan
    filtered_data = filter_data(filtered_data, street=street)

    statezip_options = get_unique_options(filtered_data, 'statezip')
    statezip = st.sidebar.selectbox('Negara Bagian dan Kode Pos', options=statezip_options, index=0)

    # Input fitur lainnya
    bedrooms = st.sidebar.number_input(
        'Jumlah Kamar Tidur',
        min_value=int(data['bedrooms'].min()),
        max_value=int(data['bedrooms'].max()),
        value=int(data['bedrooms'].median())
    )
    bathrooms = st.sidebar.number_input(
        'Jumlah Kamar Mandi',
        min_value=int(data['bathrooms'].min()),
        max_value=int(data['bathrooms'].max()),
        value=int(data['bathrooms'].median())
    )
    sqft_living = st.sidebar.number_input(
        'Luas Bangunan (sqft)',
        min_value=int(data['sqft_living'].min()),
        max_value=int(data['sqft_living'].max()),
        value=int(data['sqft_living'].median())
    )
    sqft_lot = st.sidebar.number_input(
        'Luas Tanah (sqft)',
        min_value=int(data['sqft_lot'].min()),
        max_value=int(data['sqft_lot'].max()),
        value=int(data['sqft_lot'].median())
    )
    floors = st.sidebar.slider(
        'Jumlah Lantai',
        min_value=int(data['floors'].min()),
        max_value=int(data['floors'].max()),
        value=int(data['floors'].median())
    )
    condition = st.sidebar.slider(
        'Kondisi Rumah (1-5)',
        min_value=int(data['condition'].min()),
        max_value=int(data['condition'].max()),
        value=int(data['condition'].median())
    )
    sqft_above = st.sidebar.number_input(
        'Luas Atas Tanah (sqft)',
        min_value=int(data['sqft_above'].min()),
        max_value=int(data['sqft_above'].max()),
        value=int(data['sqft_above'].median())
    )
    sqft_basement = st.sidebar.number_input(
        'Luas Basement (sqft)',
        min_value=int(data['sqft_basement'].min()),
        max_value=int(data['sqft_basement'].max()),
        value=int(data['sqft_basement'].median())
    )
    yr_built = st.sidebar.number_input(
        'Tahun Dibangun',
        min_value=int(data['yr_built'].min()),
        max_value=int(data['yr_built'].max()),
        value=int(data['yr_built'].median())
    )
    yr_renovated = st.sidebar.number_input(
        'Tahun Renovasi',
        min_value=int(data['yr_renovated'].min()),
        max_value=int(data['yr_renovated'].max()),
        value=int(data['yr_renovated'].median())
    )

    # Buat dictionary fitur
    features = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'condition': condition,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'street': street,
        'city': city,
        'statezip': statezip,
        'country': country
    }
    
    return pd.DataFrame([features])

input_df = user_input_features()

# Menampilkan fitur yang dimasukkan
st.subheader('Fitur yang Dimasukkan')
st.write(input_df)

# Memuat model
@st.cache_resource
def load_model():
    model = joblib.load('linear_regression_model.pkl')
    return model

model = load_model()

# Prediksi
if st.button('Prediksi Harga'):
    prediction = model.predict(input_df)
    st.subheader('Prediksi Harga Rumah')
    st.write(f"Harga yang diprediksi: **${prediction[0]:,.2f}**")

# Menampilkan beberapa baris dari data untuk referensi
st.subheader('Contoh Data Rumah')
st.write(data.head())