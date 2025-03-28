from lib import *

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import NuSVC

import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from yaml import safe_load



atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'
data_file = 'denoised_data.nii.gz'
events_file = "time_file.csv"

data_dir_hc = './data/HC'
out_dir_hc = './numpy_data/HC/'
sub_num_hc = 18

data_dir_schz = './data/SCHZ/'
out_dir_schz = './numpy_data/SCHZ/'
sub_num_schz = 19

is_hc = True

if is_hc:
    data_dir = data_dir_hc
    out_dir = out_dir_hc
    sub_num = sub_num_hc
else:
    data_dir = data_dir_schz
    out_dir = out_dir_schz
    sub_num = sub_num_schz

tr = 1.0    # нужно будет задать правильное значение для каждого отдельного испытуемого
window_seconds = 10



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

    # 3. Сравнение моделей
    models = {
        'NuSVC' : NuSVC(
            nu=0.3,       # Доля опорных векторов (0 < nu < 1)
            kernel='rbf', # Ядро: 'linear', 'rbf', 'poly'
            gamma='scale', # Коэффициент для ядра RBF
            random_state=42
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

    # # 4. Обучение лучшей модели (SVM показал лучший результат)
    # param_grid = {
    #     'nu': [0.3, 0.5, 0.7],
    #     'gamma': ['scale', 'auto'],
    #     'kernel': ['rbf', 'linear']
    # }

    # grid_search = GridSearchCV(NuSVC(), param_grid, cv=3, verbose=2)
    # grid_search.fit(X_scaled, labels)

    # best_model = grid_search.best_estimator_
    # print("\nЛучшие параметры:", grid_search.best_params_)
    
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
    average = False
    if average:
        matrix =  np.load(f'results/HC/area_matrix.npy')

        model = different_models(matrix)
    else: 
        true_matrix = np.load(f'results/HC/max_matrix_true.npy')
        lie_matrix = np.load(f'results/HC/max_matrix_lie.npy')
        
        labels = np.array([0] * (true_matrix.shape[0]) + [1] * (lie_matrix.shape[0]))
        
        matrix = np.concatenate((true_matrix, lie_matrix), axis=0)
        print(matrix.shape)
        print(labels.shape)
        model = different_models(matrix, labels)


    # Сохранение модели в файл
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)



