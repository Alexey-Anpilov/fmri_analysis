import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.svm import NuSVC
import pandas as pd
from sklearn.model_selection import GridSearchCV
from yaml import safe_load
import seaborn as sns
import matplotlib.pyplot as plt


def visualize(data):
# Создаем тепловую карту
    plt.figure(figsize=(10, 6))  # Задаем размер графика
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Значения'})  # 'viridis' — цветовая карта
    plt.title('Тепловая карта массива')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.show()



def different_models(matrix, labels=None):
    # 1. Загрузка данных и подготовка
    # Предположим, матрица и метки уже загружены
    # matrix.shape = (n_subjects, 132), labels.shape = (n_subjects,)
    # Пример синтетических данных:
    n_subjects = matrix.shape[0]
    if labels is None:
        labels = np.array([0] * (n_subjects//2) + [1] * (n_subjects//2))

    # Перемешивание данных
    indices = np.random.permutation(n_subjects)
    matrix, labels = matrix[indices], labels[indices]

    # 2. Масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(matrix)

    visualize(X_scaled)

    # 3. Сравнение моделей
    models = {
        'NuSVC' : NuSVC(
            nu=0.3,       # Доля опорных векторов (0 < nu < 1)
            kernel='rbf', # Ядро: 'linear', 'rbf', 'poly'
            gamma='scale', # Коэффициент для ядра RBF
            random_state=42,
            class_weight='balanced'
        ),
    }

    model = NuSVC(
            nu=0.3,       # Доля опорных векторов (0 < nu < 1)
            kernel='rbf', # Ядро: 'linear', 'rbf', 'poly'
            gamma='scale', # Коэффициент для ядра RBF
            random_state=42
        )

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, labels, cv=5, scoring='accuracy')
        print(f"{name} | Средняя точность: {scores.mean():.2f} (±{scores.std():.2f})")

    # 4. Обучение лучшей модели (SVM показал лучший результат)
    param_grid = {
        'nu': [0.3, 0.5, 0.7],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    }

    grid_search = GridSearchCV(NuSVC(), param_grid, cv=3, verbose=2)
    grid_search.fit(X_scaled, labels)

    best_model = grid_search.best_estimator_
    print("\nЛучшие параметры:", grid_search.best_params_)
    
    best_model = model

    # 5. Оценка на тестовой выборке
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=0.3, random_state=42
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    return best_model



def maximum(data):
    return np.max(data, axis=1)

def minimum(data):
    return np.min(data, axis=1)

def max_min(data):
    return np.max(data, axis=1) - np.min(data, axis=1)

def calculate_auc(data):
    return np.trapz(data, axis=1)  # Интегрируем по времени для каждого ответа и региона

funcs = {
    'max' : maximum,
    'min' : minimum,
    'max-min': max_min,
    'area' : calculate_auc
}



def read_events_from_file(file_name):
    with open('./tmp.csv', 'w') as output_file,\
        open(file_name, 'r') as input_file:
            output_file.write('onset,duration,trial_type\n')
            for line in input_file:
                l = line.split()
                if len(l) == 3:
                    continue
                if l[3][2] == '4':
                    output_file.write(l[1] + ',1.0,'  +  '1'   + '\n')
                else:
                    output_file.write(l[1] + ',1.0,' + '0' + '\n')
    events_data = pd.read_csv('./tmp.csv')

    return events_data


def process_config(config_path):
    with open(config_path) as f:
        data = safe_load(f)
        data_file = data['data_path']
        events_file = data['events_path']
        tr = data['tr']
        return data_file, events_file, tr

if __name__ == '__main__':
    average = True
    if average:
        matrix =  np.load(f'/home/aaanpilov/diploma/project/truth_lie_matrix_HC_raw.npy')
        print(matrix.shape)
        model = different_models(matrix)
    # else: 
    #     true_matrix = np.load(f'results/HC/max_matrix_true.npy')
    #     lie_matrix = np.load(f'results/HC/max_matrix_lie.npy')
        
    #     labels = np.array([0] * (true_matrix.shape[0]) + [1] * (lie_matrix.shape[0]))
        
    #     matrix = np.concatenate((true_matrix, lie_matrix), axis=0)
    #     print(matrix.shape)
    #     print(labels.shape)
    #     model = different_models(matrix, labels)

    # matrix = np.load('./results/ranks_matrix.npy')
    # N = matrix.shape[0]  # Длина массива
    # arr = np.zeros(N, dtype=int)  # Создаем массив из нулей
    # arr[3::5] = 1  # Каждый 4-й элемент (начиная с индекса 3) делаем 1
    # model = different_models(matrix, arr)
    # matrix_schz = np.load(f'results/SCHZ/area_matrix.npy')
    # print(model.predict(matrix_schz))    


    # # Сохранение модели в файл
    # with open('ranks_model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)



