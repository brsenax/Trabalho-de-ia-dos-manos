#importamos as ferramentas necessaárias para fazer o modelo preditivo e sua avaliação
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
import seaborn as sns
import matplotlib.pyplot as plt

#fazemos uma seleção de variáveis
selecao = SelectKBest(score_func=f_classif, k=10)
x_data = selecao.fit_transform(x_data, y_data)

#dividimos os dados 75% treino e 25% teste
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

#fazemos uma regressão analisando o r-quadrado
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred_reg = reg.predict(X_test)
r2 = r2_score(y_test, y_pred_reg)

print(f"R² da Regressão Linear: {r2:.3f}")


#TREINAMENTO (KNN)
# Criando o modelo com 5 vizinhos
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#AVALIAÇÃO
#Fazemos as previsões
y_pred = knn.predict(X_test)

#resultados
print(f"Acurácia Final: {accuracy_score(y_test, y_pred):.3f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

#Matriz de confusaão plotada
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
plt.title('Matriz de Confusão - Modelo KNN')
plt.show()
