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

def load_data():
    df = pd.read_csv('titanic.csv')
    return df

def main():
    st.title("Titanic Data Analysis")
    df = load_data()

    st.header("Data Overview")
    st.write(df.head())

    st.header("Null Value Analysis")
    st.write(df.isnull().sum())

    st.header("Data Cleaning")
    df['Cabin'].fillna('No_Cabin_Data', inplace=True)
    st.write("Replaced NaN values in Cabin column with 'No_Cabin_Data'")

    le = LabelEncoder()
    df['Sex1'] = le.fit_transform(df['Sex'])
    df_age_not_null = df[df['Age'].notnull()]
    df_age_is_null = df[df['Age'].isnull()]
    model = LinearRegression()
    model.fit(df_age_not_null[['Fare', 'Sex1', 'Pclass']], df_age_not_null['Age'])
    predicted_ages = model.predict(df_age_is_null[['Fare', 'Sex1', 'Pclass']])
    df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
    st.write("Replaced NaN values in Age column with predictions using linear regression")

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    st.write("Replaced NaN values in Embarked column with the most frequent value")

    st.header("Data Visualization")
    fig, ax = plt.subplots()
    df['Survived'].value_counts().plot.pie(
        colors=('tab:red', 'tab:blue'),
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
    st.write(df['Title'].value_counts())

    st.header("Survival Rate by Title")
    def agrupar_titulos(titulo):
        if titulo in ['Mr', 'Miss', 'Mrs']:
            return titulo
        else:
            return 'Otros'
    df['Grupo_Titulo'] = df['Title'].apply(agrupar_titulos)
    fig, ax = plt.subplots()
    df.groupby(['Grupo_Titulo', 'Survived']).size().unstack(fill_value=0).plot(kind='bar', ax=ax)
    ax.set_title('Supervivientes y No Supervivientes por Título')
    ax.set_xlabel('Título')
    ax.set_ylabel('Cantidad')
    ax.legend(['No Sobrevivió', 'Sobrevivió'])
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
