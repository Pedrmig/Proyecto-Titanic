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
        options = ["Home","Datos","Análisis","Filtros"],
        icons = ["house","book","bar-chart","filter"],
        menu_icon = "cast",
        default_index = 0,)
    if selected == "Home":
        st.title(f"{selected}")
        st.header('Análisis del conjunto de datos del Titanic para identificar factores que influyeron en la supervivencia de los pasajeros.')
        #st.image('Images/titanic_Underwater.png',use_column_width=True)

    if selected == "Datos":
        st.title(f"{selected}")
        st.header('Se ha usado Python y Jupiter Notebook para el análisis.')
        st.subheader('Librerías utilizadas: Pandas, Numpy, Matplotlib, Seaborn, Sklearn, IPython, Streamlit') 
        
    if selected == "Análisis":
        st.title(f"{selected}")
        st.header('Análisis de datos del Titanic y representación visual de los resultados.')
        
    if selected == "Filtros":
        st.title(f"{selected}")
        st.header('Haga sus proprios filtros y consultas.')
    
# creando el contenido de las páginas de acuerdo a la opción seleccionada
if selected == "Home": 
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Análisis de datos del Titanic</h1>", unsafe_allow_html=True)
    st.write('El análisis de datos es un proceso que se utiliza para inspeccionar, limpiar y modelar datos con el objetivo de descubrir información útil, llegar a conclusiones y apoyar la toma de decisiones.')
    st.write('El conjunto de datos del Titanic contiene información sobre los pasajeros del Titanic, incluidos detalles como la edad, el sexo, la clase de pasajero, el puerto de embarque y si sobrevivieron o no.')
    st.write('El análisis de datos del Titanic se centra en identificar factores que influyeron en la supervivencia de los pasajeros.')
    st.image('Images/titanic_real.png',use_column_width=True) 
    
if selected == "Datos":

    st.title("Analise de Datos del Titanic")
    df = pd.read_csv('titanic.csv')

    st.header("Data Overview")
    st.write(df.head())

    st.header("Analisis de valores nulos")
    st.write(((df.isnull().sum() / len(df))*100).to_frame().T)

    st.header("Limpieza de los Datos")

    st.markdown("<h1 style='text-align: left; color: white; font-size:20px;'>Reemplazando los valores NaN en la columna Cabin con ‘No_Cabin_Data'</h1>", unsafe_allow_html=True)
    st.write("df['Cabin'].fillna('No_Cabin_Data', inplace=True)")
    df['Cabin'].fillna('No_Cabin_Data', inplace=True)

    st.markdown("<h1 style='text-align: left; color: white; font-size:20px;'>Reemplazando los valores NaN en la columna de Edad con predicciones usando regresión lineal", unsafe_allow_html=True)
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

    st.markdown("<h1 style='text-align: left; color: white; font-size:20px;'>Reemplazando los valores NaN en la columna Embarked con el valor más frecuente", unsafe_allow_html=True)
    st.write("df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)")
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    st.header("Diccionario de Datos")
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
    mapa_embarked = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    df['Embarked'] = df['Embarked'].replace(mapa_embarked)
    
    # haciendo el Data Visualization
    st.header("Tasa de Supervivencia")
    fig, ax = plt.subplots()
    df['Survived'].value_counts().plot.pie(
        colors=('tab:orange', 'tab:blue'),
        title='Tarifa de Supervivencia',
        fontsize=13,
        shadow=True,
        startangle=90,
        autopct='%1.1f%%',
        labels=('Fallecidos', 'Supervivientes'),
        ax=ax
    )
    st.pyplot(fig)

    st.header("Tasa de Supervivencia por Clase y Sexo")
    fig, ax = plt.subplots(2, 1)
    df.groupby('Pclass')['Survived'].mean().plot(kind='bar', ax=ax[0])
    df.groupby('Sex')['Survived'].mean().plot(kind='bar', ax=ax[1])
    st.pyplot(fig)

    st.header("Tasa de Supervivencia por Grupo de Edad")
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

    st.header("Titulo de los Pasajeros")
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
    st.write(df[['Name', 'Title']].head())
    st.write(df['Title'].value_counts().to_frame().T)

    st.header("Tasa de Supervivencia por Título")
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
    
if selected == "Filtros":
    # Carga los dados una unica vez para evitar recargarlos en cada consulta
    @st.cache_data
    def cargar_datos():
        df = pd.read_csv('titanic.csv')
        return df
    df = cargar_datos()

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
    mapa_embarked = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    df['Embarked'] = df['Embarked'].replace(mapa_embarked)
    
    # Barra de consulta en la página principal
    st.header('Consulta del Usuario')
    
    # Búsqueda
    entrada_busqueda = st.text_input('Búsqueda de Nombre', '')
    mascara = df['Name'].str.contains(entrada_busqueda, case=False)
    df = df[mascara]

    if st.button('Limpiar'):
        df = cargar_datos()
    
    # Principal
    st.header('Datos del Titanic')
    st.write(df)

    # Selección de consulta
    st.header('Personalize tu consulta')
    consulta = st.selectbox('', ['','Hombres y Mujeres', 'Por Edad', 'Por Embarque', 'Por Clase'], key='1')

    if consulta == 'Hombres y Mujeres':
        hombres = df[df['Sex'] == 'male']['Survived'].value_counts()
        mujeres = df[df['Sex'] == 'female']['Survived'].value_counts()
        st.write('Hombres - Vivos: ', hombres[1], ' Muertos: ', hombres[0])
        st.write('Mujeres - Vivas: ', mujeres[1], ' Muertas: ', mujeres[0])
        fig, ax = plt.subplots()
        ax.bar(['H - Vivos', 'H - Muertos', 'M - Vivas', 'M - Muertas'], [hombres[1], hombres[0], mujeres[1], mujeres[0]], color=['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange'])
        st.pyplot(fig)

    elif consulta == 'Por Edad':
        edad = st.slider('Elige la edad', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=int(df['Age'].mean()))
        vivos = df[(df['Age'] >= edad) & (df['Survived'] == 1)].shape[0]
        muertos = df[(df['Age'] >= edad) & (df['Survived'] == 0)].shape[0]
        st.write('Vivos mayores de ', edad, ' años: ', vivos)
        st.write('Muertos mayores de ', edad, ' años: ', muertos)
        fig, ax = plt.subplots()
        ax.bar(['Sobrevivientes mayores de ' + str(edad), 'Fallecidos mayores de ' + str(edad)], [vivos, muertos], color=['tab:blue', 'tab:orange'])
        st.pyplot(fig)
        
    elif consulta == 'Por Embarque':
        embarque = st.selectbox('Elige el lugar de embarque', df['Embarked'].unique(), key='2')
        vivos = df[(df['Embarked'] == embarque) & (df['Survived'] == 1)].shape[0]
        muertos = df[(df['Embarked'] == embarque) & (df['Survived'] == 0)].shape[0]
        st.write('Vivos que embarcaron en ', embarque, ': ', vivos)
        st.write('Muertos que embarcaron en ', embarque, ': ', muertos)
        fig, ax = plt.subplots()
        ax.bar(['Sobrevivientes de ' + embarque, 'Fallecidos de' + embarque], [vivos, muertos], color=['tab:blue', 'tab:orange'])
        st.pyplot(fig)
    
    elif consulta == 'Por Clase':
        clase = st.selectbox('Elige la clase', df['Pclass'].unique(), key='3')
        vivos = df[(df['Pclass'] == clase) & (df['Survived'] == 1)].shape[0]
        muertos = df[(df['Pclass'] == clase) & (df['Survived'] == 0)].shape[0]
        st.write('Vivos de la clase ', clase, ': ', vivos)
        st.write('Muertos de la clase ', clase, ': ', muertos)
        fig, ax = plt.subplots()
        ax.bar(['Vivos de la clase ' + str(clase), 'Muertos de la clase ' + str(clase)], [vivos, muertos], color=['tab:blue', 'tab:orange'])
        st.pyplot(fig)
        
    if st.button('Limpar'):
        df = cargar_datos()
