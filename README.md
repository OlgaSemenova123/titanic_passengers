В работе прогнозируется выживет ли человек на Титанике по заданным характеристикам. Данные берутся с Kaggle. На тестовых данных от Kaggle модель предсказала 77%. Алгоритм - дерево решений

В переменные не были включены следующие столбцы:
X = titanic_data.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

Пустые значения были заполнены средней по полу:
age_female_median = X[X.Sex_female == True].Age.median()

age_male_median = X[X.Sex_male == True].Age.median()

X[X.Sex_female == True] = X[X.Sex_female == True].fillna({'Age': age_female_median})

X[X.Sex_male == True] = X[X.Sex_male == True].fillna({'Age': age_male_median})

При расчете были выбраны следующие оптимальные параметры:
{'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 4}

Общая оценка модели, а так же Precision, Recall
best_clf.score: 0.8203389830508474
precision_score_first: 0.8085106382978723
recall_score_first: 0.6846846846846847

Посмотрим на график вероятностей отнесения к классам "выживших" и "мертвых". Изначально он стоит 0,5, выше этой цифры модель относит к "выжившим"
![Figure_1](https://github.com/OlgaSemenova123/titanic_passengers/assets/157280225/058f8e72-2b5a-481d-a541-c67415f5fc43)
Тут можно менять данные, но видно, что от 0,7 отдельная группа, можно установить именно эту отметку отсечения. Для принятия решения нужно еще посмотреть ROC-кривую
![Figure_2](https://github.com/OlgaSemenova123/titanic_passengers/assets/157280225/b71f8b24-f95d-49ca-875f-574aa1f0494d)
Нам нужно максимизировать True Positive Rate и брать минимальное False Positive Rate, 0,7 здесь так же подходит, при ней False Rate будет на отметке примерно 0,1.

Это в свою очередь увеличит Precision и уменьшит Recall. Посмотрим на результат:
precision_score_second: 0.8292682926829268
recall_score_second: 0.6126126126126126
