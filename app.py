import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit_option_menu
import importlib.util
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder



st.set_page_config(page_title='Análisis Tytanic Dataset' ,layout="wide",page_icon='Boat')

with st.sidebar:
  selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home","Datos","Análisis"],
    icons = ["house","book","bar-chart"],
    menu_icon = "cast",
    default_index = 0,)

  if selected == "Home":
    st.title(f"{selected}")
    st.header('Análisis del conjunto de datos del Titanic para identificar factores que influyeron en la supervivencia de los pasajeros.')
    st.image('Images/titanic_Underwater.jpg',use_column_width=True)
     
  if selected == "Datos":
    st.title(f"{selected}")
    st.header('Se ha usado Python y Jupiter Notebook para el análisis.')
    st.subheader('Librerías utilizadas: Pandas, Numpy, Matplotlib, Seaborn, Sklearn, IPython, Streamlit') 
        
  if selected == "Análisis":
    st.title(f"{selected}")
    st.header('Análisis de datos del Titanic')
    
    
if selected == "Home": 
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Análisis de datos del Titanic!</h1>", unsafe_allow_html=True)
    st.write('El análisis de datos es un proceso que se utiliza para inspeccionar, limpiar y modelar datos con el objetivo de descubrir información útil, llegar a conclusiones y apoyar la toma de decisiones.')
    st.write('El conjunto de datos del Titanic contiene información sobre los pasajeros del Titanic, incluidos detalles como la edad, el sexo, la clase de pasajero, el puerto de embarque y si sobrevivieron o no.')
    st.write('El análisis de datos del Titanic se centra en identificar factores que influyeron en la supervivencia de los pasajeros.')
    st.image('Images/titanic_real.jpeg',use_column_width=True)
    
if selected == "Datos":
   st.image('Images/data_dic.png',use_column_width=True)
     