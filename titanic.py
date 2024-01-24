import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score
pd.set_option('display.max_columns', 500)
np.random.seed(0)

# выбираем переменные
titanic_data = pd.read_csv('/home/andrey4281/Загрузки/train.csv')
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

# Превращаем строковые столбцы в числовые
X = pd.get_dummies(X)

# Заполняем пустые значения средней по полу
age_female_median = X[X.Sex_female == True].Age.median()
age_male_median = X[X.Sex_male == True].Age.median()
X[X.Sex_female == True] = X[X.Sex_female == True].fillna({'Age': age_female_median})
X[X.Sex_male == True] = X[X.Sex_male == True].fillna({'Age': age_male_median})

# Разбиваем на train и test, подбираем оптимальные параметры, строим дерево решений
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
parametrs = {'criterion': ['gini', 'entropy'], 'max_depth': range(1, 10), 'min_samples_leaf': range(2, 10),
             'min_samples_split': range(2, 10)}
clf = tree.DecisionTreeClassifier()

grid_search_cv_clf = GridSearchCV(clf, parametrs, cv=5, scoring='f1')
grid_search_cv_clf.fit(X_train, y_train)
print(grid_search_cv_clf.best_params_)

# Обучаем и предсказываем по выбранным параметрам, оцениваем метрики precision, recall
best_clf = grid_search_cv_clf.best_estimator_
best_clf.fit(X_train, y_train)
print('best_clf.score:', best_clf.score(X_test, y_test))
y_pred = best_clf.predict(X_test)
print('precision_score_first:', precision_score(y_test, y_pred))
print('recall_score_first:', recall_score(y_test, y_pred))

# Посмотрим вероятность отнесения к "выжившим"
y_predicted_prob = best_clf.predict_proba(X_test)
pd.Series(y_predicted_prob[:, 1]).hist()
plt.show()

# Изменим вероятность для оптимизации precision
y_pred = np.where(y_predicted_prob[:, 1] > 0.7, 1, 0)
print('precision_score_second:', precision_score(y_test, y_pred))
print('recall_score_second:', recall_score(y_test, y_pred))
# tree.plot_tree(best_clf)

# Загрузим тестовые данные для отправки на kaggle
titanic_data = pd.read_csv('/home/andrey4281/Загрузки/test.csv')
X_testing = titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X_testing = pd.get_dummies(X_testing)
X_testing[X_testing.Sex_female == True] = X_testing[X_testing.Sex_female == True].fillna({'Age': age_female_median})
X_testing[X_testing.Sex_male == True] = X_testing[X_testing.Sex_male == True].fillna({'Age': age_male_median})
X_testing = X_testing.fillna(0)
result = pd.Series(best_clf.predict(X_testing))
result_predicted_prob = best_clf.predict_proba(X_testing)
result = pd.Series(np.where(result_predicted_prob[:, 1] > 0.7, 1, 0))
final = pd.DataFrame()
final = pd.concat([titanic_data['PassengerId'], result], axis=1)
final.columns = ['PassengerId', 'Survived']
final.to_csv(r'/home/andrey4281/Загрузки/prediction.csv', index=False)

# Дополнительно построим ROC-кривую для отбора вероятности выбора класса
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])
roc_auc= auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()








