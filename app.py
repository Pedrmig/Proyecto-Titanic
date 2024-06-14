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

# creando el menú de opciones
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
    
# creando el contenido de las páginas de acuerdo a la opción seleccionada
if selected == "Home": 
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Análisis de datos del Titanic</h1>", unsafe_allow_html=True)
    st.write('El análisis de datos es un proceso que se utiliza para inspeccionar, limpiar y modelar datos con el objetivo de descubrir información útil, llegar a conclusiones y apoyar la toma de decisiones.')
    st.write('El conjunto de datos del Titanic contiene información sobre los pasajeros del Titanic, incluidos detalles como la edad, el sexo, la clase de pasajero, el puerto de embarque y si sobrevivieron o no.')
    st.write('El análisis de datos del Titanic se centra en identificar factores que influyeron en la supervivencia de los pasajeros.')
    st.image('Images/titanic_real.jpeg',use_column_width=True)
    
if selected == "Datos":

    st.title("Titanic Data Analysis")
    df = pd.read_csv('titanic.csv')

    st.header("Data Overview")
    st.write(df.head())

    st.header("Null Value Analysis")
    st.write(df.isnull().sum().to_frame().T)

    st.header("Data Cleaning")

    st.markdown("<h1 style='text-align: left; color: white; font-size:20px;'>Replaced NaN values in Cabin column with 'No_Cabin_Data'</h1>", unsafe_allow_html=True)
    st.write("df['Cabin'].fillna('No_Cabin_Data', inplace=True)")
    df['Cabin'].fillna('No_Cabin_Data', inplace=True)

    st.markdown("<h1 style='text-align: left; color: white; font-size:20px;'>Replaced NaN values in Age column with predictions using linear regression", unsafe_allow_html=True)
    st.write("le = LabelEncoder()")
    st.write("df['Sex1'] = le.fit_transform(df['Sex'])")
    st.write("df_age_not_null = df[df['Age'].notnull()]")
    st.write("df_age_is_null = df[df['Age'].isnull()]")
    st.write("model = LinearRegression()")
    st.write("model.fit(df_age_not_null[['Fare', 'Sex1', 'Pclass']], df_age_not_null['Age'])")
    st.write("predicted_ages = model.predict(df_age_is_null[['Fare', 'Sex1', 'Pclass']])")
    st.write("df.loc[df['Age'].isnull(), 'Age'] = predicted_ages")
    le = LabelEncoder()
    df['Sex1'] = le.fit_transform(df['Sex'])
    df_age_not_null = df[df['Age'].notnull()]
    df_age_is_null = df[df['Age'].isnull()]
    model = LinearRegression()
    model.fit(df_age_not_null[['Fare', 'Sex1', 'Pclass']], df_age_not_null['Age'])
    predicted_ages = model.predict(df_age_is_null[['Fare', 'Sex1', 'Pclass']])
    df.loc[df['Age'].isnull(), 'Age'] = predicted_ages

    st.markdown("<h1 style='text-align: left; color: white; font-size:20px;'>Replaced NaN values in Embarked column with the most frequent value", unsafe_allow_html=True)
    st.write("df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)")
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    st.header("Data Dictionary")
    st.image('Images/data_dic.png',use_column_width=True) 
    
if selected == "Análisis":
    # haciendo el Data Cleaning
    df = pd.read_csv('titanic.csv')
    df['Cabin'].fillna('No_Cabin_Data', inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    le = LabelEncoder()
    df['Sex1'] = le.fit_transform(df['Sex'])
    df_age_not_null = df[df['Age'].notnull()]
    df_age_is_null = df[df['Age'].isnull()]
    model = LinearRegression()
    model.fit(df_age_not_null[['Fare', 'Sex1', 'Pclass']], df_age_not_null['Age'])
    predicted_ages = model.predict(df_age_is_null[['Fare', 'Sex1', 'Pclass']])
    df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
    
    # haciendo el Data Visualization
    st.header("Survival Rate")
    fig, ax = plt.subplots()
    df['Survived'].value_counts().plot.pie(
        colors=('tab:orange', 'tab:blue'),
        title='Survival Rate',
        fontsize=13,
        shadow=True,
        startangle=90,
        autopct='%1.1f%%',
        labels=('Did not survive', 'Survived'),
        ax=ax
    )
    st.pyplot(fig)

    st.header("Survival Rate by Passenger Class and Sex")
    fig, ax = plt.subplots(2, 1)
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=ax[0])
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax[1])
    st.pyplot(fig)

    st.header("Survival Rate by Age Group")
    df['Ninos_adultos'] = None
    df.loc[df[df['Age'] < 18].index, 'Ninos_adultos'] = 'Niños'
    df.loc[df[(df['Age'] >= 18) & (df['Age'] < 60)].index, 'Ninos_adultos'] = 'Adultos'
    df.loc[df[df['Age'] > 60].index, 'Ninos_adultos'] = 'Mayores'
    fig, ax = plt.subplots()
    df.groupby('Ninos_adultos')['Survived'].mean().loc[['Niños', 'Adultos', 'Mayores']].plot.barh(
        title='Promedio de Supervivencia', 
        figsize=(10, 3),
        color= ['darkblue', 'blue', 'lightblue'],
        ax=ax
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))
    st.pyplot(fig)

    st.header("Passenger Titles")
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
    st.write(df[['Name', 'Title']].head())
    st.write(df['Title'].value_counts().to_frame().T)

    st.header("Survival Rate by Title")
    def agrupar_titulos(titulo):
        if titulo in ['Mr', 'Miss', 'Mrs']:
            return titulo
        else:
            return 'Otros'
    df['Grupo_Titulo'] = df['Title'].apply(agrupar_titulos)
    fig, ax = plt.subplots()
    df.groupby(['Grupo_Titulo', 'Survived']).size().unstack(fill_value=0).plot(kind='bar', ax=ax, color=['tab:orange', 'tab:blue'])
    ax.set_title('Supervivientes y No Supervivientes por Título')
    ax.set_xlabel('Título')
    ax.set_ylabel('Cantidad')
    ax.legend(['No Sobrevivió', 'Sobrevivió'])
    
    st.pyplot(fig)