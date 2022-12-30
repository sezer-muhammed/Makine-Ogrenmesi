import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

wine_dataset_path = r"algoritmalar\regresyon\winequality-red.xlsx"
wine_dataset = pd.read_excel(wine_dataset_path)

Y = wine_dataset['quality']
X = wine_dataset.drop('quality', axis=1)

poly_model_regression = PolynomialFeatures(degree=4)
poly_X = poly_model_regression.fit_transform(X)

linear_model_regression = linear_model.LinearRegression()
linear_model_regression.fit(poly_X, Y)

print("Sabit Değer: ", linear_model_regression.intercept_)
print("Katsayılar: ", linear_model_regression.coef_)
print("R^2 Değeri: ", linear_model_regression.score(poly_X, Y))

wine_quality_prediction = linear_model_regression.predict(poly_X)
real_wine_quality = Y

print("Gerçek Değerler: ", real_wine_quality)
print("Tahmin Edilen Değerler: ", wine_quality_prediction)

plt.scatter(real_wine_quality, wine_quality_prediction)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.show()

result_dataframe = pd.DataFrame({'Gerçek Değerler': real_wine_quality, 'Tahmin Edilen Değerler': wine_quality_prediction})
result_dataframe.to_excel(r"algoritmalar\regresyon\poly_regresyon_result_degree4.xlsx")