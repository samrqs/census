import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('census.csv')

x = df.iloc[:, 0:14].values #tratamento entrada
y = df.iloc[:, 14].values #tratamento saída
#print(x)
#print(y)

#dados categóricos(sting, number, date) para indexes numericos
le_workclass = LabelEncoder()
le_education = LabelEncoder()
le_marital = LabelEncoder()
le_occupation = LabelEncoder()
le_relationship = LabelEncoder()
le_race = LabelEncoder()
le_sex = LabelEncoder()
le_country = LabelEncoder()

x[:, 1] = le_workclass.fit_transform(x[:, 1])
x[:, 3] = le_education.fit_transform(x[:, 3])
x[:, 5] = le_marital.fit_transform(x[:, 5])
x[:, 6] = le_occupation.fit_transform(x[:, 6])
x[:, 7] = le_relationship.fit_transform(x[:, 7])
x[:, 8] = le_race.fit_transform(x[:, 8])
x[:, 9] = le_sex.fit_transform(x[:, 9])
x[:, 13] = le_country.fit_transform(x[:, 13])

#print(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)

#print(x)
#80% das linhas vão ser para treino
#20% das linhas vão ser para teste (teste_size)
x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.20) 

print(x_teste.shape, x_teste.shape)

p = 10
pca = PCA(n_components=p)

#definimos novas vaiaveis de treino e teste (com menor numero de colunas)
x_treino_pca = pca.fit_transform(x_treino)
x_teste_pca = pca.transform(x_teste)

#print(x_teste_pca.shape, x_teste_pca.shape)

pca.explained_variance_ratio_

total_variancia = pca.explained_variance_ratio_.sum()
print(f'O modelo PCA com {p} váriaveis explica {100*(total_variancia): .2f} % dos dados de entrada')

modelo_rf = RandomForestClassifier(n_estimators=40, random_state=0)

modelo_rf.fit(x_treino, y_treino)
previsoes = modelo_rf.predict(x_teste)

print(previsoes, y_teste)

acuracia = accuracy_score(y_teste, previsoes)
print(f'A acurácia do modelo Random Forest é de: {(100*acuracia): .2f} %')