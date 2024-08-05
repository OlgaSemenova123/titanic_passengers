# titanic_passengers
Участие в известном ML соревновании на kaggle по предсказанию пассажиров, которые выживут: https://www.kaggle.com/competitions/titanic

Для начала отберем параметры для обучения модели. Удалим такие колонки, как 'PassengerId', 'Name', 'Ticket', 'Cabin'.
Заполним медианами пропущенные значения по возрасту отдельно для мужчин и женщин:

age_female_median = X[X.Sex_female == True].Age.median()

age_male_median = X[X.Sex_male == True].Age.median()

X[X.Sex_female == True] = X[X.Sex_female == True].fillna({'Age': age_female_median})

X[X.Sex_male == True] = X[X.Sex_male == True].fillna({'Age': age_male_median})

Через жадный алогритм построим дерево решений. Будем подбирать параметры, такие как критерий, глубина дерева, минимальное количество листьев и разделение.
В итоге получаем следующие параметры:

{'criterion': 'gini', 'max_depth': 8, 'min_samples_leaf': 5, 'min_samples_split': 4}

Оценим метрики precision и recall:

precision_score_first: 0.81

recall_score_first: 0.68

Посмотрим на вероятность отнесения к выжившим:
![Figure_1](https://github.com/user-attachments/assets/baa384ba-8085-4584-bd76-729cee7f8994)

По дефолту стоит 0,5 разделение, но логичнее бы было взять по 0,7. Попробуем так же взглянуть на ROC-кривую
![Figure_2](https://github.com/user-attachments/assets/00018d86-1b21-4847-81f3-43ebcb235248)

Максимум тут достигается при 0,7-0,8, но т к мы не хотим сильно уменьшать recall, то остановимся на 0,7.
При изменении вероятности предсказания получаем следующие precision и recall:

precision_score_second: 0.83

recall_score_second: 0.61

Видно, что увеличился precision, и упал recall, предсказываем тестовую выборку. По оценке на kaggle решение верно на 77%.
