# Linear-regression

Решим линейную регрессию при помощи МНК
Метод наименьших квадратов (МНК, англ. Ordinary Least Squares, OLS) — математический метод, применяемый для решения различных задач, основанный на минимизации суммы квадратов отклонений некоторых функций от искомых переменных. 

![](https://raw.githubusercontent.com/okiochan/Linear-regression/master/formula/f1.gif)

На python есть готовый метод для решения МНК линейной регрессии

```
np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
```
Воспользуемся им и применим ридж регуляризацию, чтобы гарантировано взялась обрантая матрица с коэф-ом С = 1e-7. 
В методе: Х - вектор(матрица) параметров в линейной регрессии, у - вектор ответов для Х, С - коэф регуляризации.

```
def RidgeRegression(X,y,C):
    l = X.shape[0]
    n = X.shape[1]

    # bias trick - concatenate ones in front of matrix
    ones = np.atleast_2d(np.ones(l)).T
    X = np.concatenate((ones,X),axis=1)

    # learn linear MNK
    res = np.linalg.inv(X.T.dot(X) + np.eye(n+1) * C).dot(X.T.dot(y))
    return res[1:(n+1)], res[0]
```

Для оценки качества решения, будем брать **SSE**
Выборку возьмем реальную **plant_with_control** и разобьем ее на тестируемую и контрольную

Также сравним качество регрессии, с применением [PCA]( https://github.com/okiochan/PCA)
Возьмем одну компоненту (выборка такая, что мы не потеряем много информации, взяв одну компоненту)

посмотрим на результаты

![](https://raw.githubusercontent.com/okiochan/Linear-regression/master/img/i1.png)

![](https://raw.githubusercontent.com/okiochan/Linear-regression/master/img/i2.png)
