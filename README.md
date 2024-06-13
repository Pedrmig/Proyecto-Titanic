# Titanic Survival Analysis

## Introduction
This project focuses on analyzing the Titanic dataset to identify factors influencing passenger survival. 

### Data Reading
The data is loaded using pandas:
```python
import pandas as pd
df = pd.read_csv('titanic.csv')
```
## Data Cleaning
The ‘Age’ and ‘Cabin’ columns contain a significant number of null entries, necessitating careful data cleaning.

#### Constult
Initiating query for null records.
```python
df.isnull().sum() / len(df) * 100 
```

#### Cabin column
Replacing the NaN values in the 'Cabin' column with 'No_Cabin_Data'
```python
df['Cabin'].fillna('Sin_Datos_Cabina', inplace=True)
```

#### Age column
Substituting NaN values in the ‘Age’ column with predictions using linear regression.
```python
le = LabelEncoder()
df['Sex1'] = le.fit_transform(df['Sex'])
df_age_not_null = df[df['Age'].notnull()]
df_age_is_null = df[df['Age'].isnull()]
model = LinearRegression()
model.fit(df_age_not_null[['Fare', 'Sex1', 'Pclass']], df_age_not_null['Age'])
predicted_ages = model.predict(df_age_is_null[['Fare', 'Sex1', 'Pclass']])
df.loc[df['Age'].isnull(), 'Age'] = predicted_ages
```

#### Gender Impact
Analysis revealed a significant correlation between gender and survival:
```python
df.groupby('Sex')['Survival'].mean().plot(kind='bar')
```

#### Age Impact
Age's role was explored, highlighting varying survival rates across age groups:
```python
# Divide los datos en grupos de niños y adultos
df['Ninos_adultos'] = None
df.loc[df[df['Age'] < 18].index, 'Ninos_adultos'] = 'Niños'
df.loc[df[(df['Age'] >= 18) & (df['Age'] < 60)].index, 'Ninos_adultos'] = 'Adultos'
df.loc[df[df['Age'] > 60].index, 'Ninos_adultos'] = 'Mayores'

# Función para formatear el eje x en porcentaje
formatter = FuncFormatter(lambda x, _: f'{x:.0%}')

# Genera gráfico de supervivencia de niños y adultos en porcentaje
ordem = ['Niños', 'Adultos', 'Mayores']
df.groupby('Ninos_adultos')['Survived'].mean().loc[ordem].plot.barh(
    title='Promedio de Supervivencia', 
    figsize=(10, 3),
    color= ['darkblue', 'blue', 'lightblue'],
).set_ylabel('')

# Aplicar el formato de porcentaje al eje x
plt.gca().xaxis.set_major_formatter(formatter)

# Establecer la etiqueta del eje x
plt.xlabel('')

# Mostrar el gráfico
plt.show()
```

#### Title Impact
Analysis revealed a significant correlation between title and survival:
```python
# Función para agrupar los títulos
def agrupar_titulos(titulo):
    if titulo in ['Mr', 'Miss', 'Mrs']:
        return titulo
    else:
        return 'Otros'

# Aplicando la función para agrupar los títulos
df['Grupo_Titulo'] = df['Title'].apply(agrupar_titulos)

# Agrupando los datos por título y supervivencia
agrupado = df.groupby(['Grupo_Titulo', 'Survived']).size().unstack(fill_value=0)

# Generando el gráfico
agrupado.plot(kind='bar')
plt.title('Supervivientes y No Supervivientes por Título')
plt.xlabel('Título')
plt.ylabel('Cantidad')
plt.legend(['No Sobrevivió', 'Sobrevivió'])
plt.show()
```

## Conclusion
The analysis highlighted key factors like gender, class, title and age in survival prediction.

## Future Work
Future enhancements could include integrating more features, trying advanced models, and using larger datasets for more robust predictions.

## How to Run the Notebook
1. Ensure Python and Jupyter Notebook are installed.
2. Install necessary packages: `pandas`, `sklearn`, `matplotlib`, `tensorflow`.
3. Run the notebook cell by cell to observe each step of the analysis and modeling.
