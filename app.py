import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
import pickle 

#import model 
dt = pickle.load(open('DT.pkl','rb'))

#load dataset
data = pd.read_excel('Price_Range_Phone_Dataset.xlsx')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Kisaran Harga Ponsel')

html_layout1 = """
<br>
<div style="background-color:green ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Kisaran Harga Ponsel</b></h2>
</div>
<br>
<br>
"""

background_image = """
<style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://cdns.klimg.com/merdeka.com/i/w/news/2022/08/10/1461428/540x270/4-teknologi-smartphone-terkini-yang-wajib-dimiliki-bagi-yang-mau-ganti-gadget-baru.jpg");
        background-size: 100vw 100vh;
        background-position: center;
        background-repeat: no-repeat;
        background-color: rgba(255, 255, 255, 0.5);
    }
}
"""

st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Decision Tree','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Ponsel')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Kisaran Harga Ponsel</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDA'):
    pr =ProfileReport(data,explorative=True)
    st.header('Input Dataframe')
    st.write(data)
    st.write('---')
    st.header('Profiling Report')
    st_profile_report(pr)

#train test split
X = data.drop('price_range',axis=1)
y = data['price_range']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    battery_power = st.sidebar.slider('Kapasitas Baterai',0,200,108)
    blue = st.sidebar.slider('Bluetooth',0,1,0)
    clock_speed = st.sidebar.slider('Kecepatan Ponsel',0.5,2.5,1.2)
    dual_sim = st.sidebar.slider('Dual SIM',0,1,0)
    fc = st.sidebar.slider('MP Kamera Depan',0,15,1)
    four_g = st.sidebar.slider('4G',0,1,0)
    int_memory = st.sidebar.slider('Memory Internal',0,130,64)
    m_dep = st.sidebar.slider('Kedalaman Ponsel',0.10,1.0,0.9)
    mobile_wt = st.sidebar.slider('Berat Ponsel',0,200,100)
    n_cores = st.sidebar.slider('Jumlah Core',0,8,4)
    pc = st.sidebar.slider('MP Kamera Utama',0,20,6)
    px_height = st.sidebar.slider('Tinggi Resolusi',0,2000,1000)
    px_width = st.sidebar.slider('Lebar Resolusi',0,2000,1000)
    ram = st.sidebar.slider('RAM',0,4000,2000)
    sc_h = st.sidebar.slider('Tinggi Layar',0,20,15)
    sc_w = st.sidebar.slider('Lebar Layar',0,20,15)
    talk_time = st.sidebar.slider('Waktu Terlama',0,20,10)
    three_g = st.sidebar.slider('3G',0,1,0)
    touch_screen = st.sidebar.slider('Layar Sentuh',0,1,0)
    wifi = st.sidebar.slider('Wifi',0,1,0)


    user_report_data = {
        'battery_power':battery_power,
        'blue':blue,
        'clock_speed':clock_speed,
        'dual_sim':dual_sim,
        'fc':fc,
        'four_g':four_g,
        'int_memory':int_memory,
        'm_dep':m_dep,
        'mobile_wt':mobile_wt,
        'n_cores':n_cores,
        'pc':pc,
        'px_height':px_height,
        'px_width':px_width,
        'ram':ram,
        'sc_h':sc_h,
        'sc_w':sc_w,
        'talk_time':talk_time,
        'three_g':three_g,
        'touch_screen':touch_screen,
        'wifi':wifi,
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Ponsel yang dipilih :')
st.write(user_data)

user_result = dt.predict(user_data)
dt_score = accuracy_score(y_test,dt.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Murah'
elif user_result[0]==1:
    output='Biasa'
elif user_result[0]==2:
    output='Mahal'

st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(dt_score*100)+'%')