import logging
from fmri_processing import *
from fmri_processing.functions import *
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import recall_score, make_scorer
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.metrics import fbeta_score, make_scorer
from fmri_processing import *
# Баллы
def normalize(data):
# Ваш массив (5 стимулов × 132 региона)

    # Для каждого региона (столбца) получаем ранги стимулов
    ranks = np.zeros_like(data, dtype=int)

    for region in range(data.shape[1]):
        # Получаем индексы, которые сортируют значения в столбце (от меньшего к большему)
        sorted_indices = np.argsort(data[:, region])
        # Преобразуем индексы в ранги (1 для минимального, 6 для максимального)
        ranks[sorted_indices, region] = np.arange(1, 6)  # 1..6
    return ranks

# Баллы 1-2
def normalize_reduced(data):
    # Создаем массив нулей того же размера, что и входные данные
    ranks = np.zeros_like(data, dtype=int)
    
    for region in range(data.shape[1]):
        # Получаем индексы, которые сортируют значения в столбце (от меньшего к большему)
        sorted_indices = np.argsort(data[:, region])
        # Присваиваем:
        # - 0 для всех элементов по умолчанию (уже сделано при создании ranks)
        # - 1 для второго по величине
        ranks[sorted_indices[-2], region] = 1
        # - 2 для максимального
        ranks[sorted_indices[-1], region] = 2
    
    return ranks


class FeatureMatrixBuilder:
    def __init__(self):
        self.logger = self._setup_logger()


    def _setup_logger(self):
        """Инициализирует и настраивает логгер для класса"""
        logger = logging.getLogger("FeatureMatrixBuilder")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
            )
            logger.addHandler(handler)
        return logger
    

    def build_matrix(self, config, subjects, processing_func=calc_maximum, normalize_func=normalize):
        feature_type = config['features_type']
        if feature_type != 'average_stimulus':
            if feature_type == 'ranks_five':
                normalize_func = normalize
            elif feature_type == 'ranks_two':
                normalize_func = normalize_reduced
            else:
                # TODO: throw exception
                print('Unknown type')
            return self.build_matrix_by_ranks(config, subjects, processing_func, normalize_func)
        else:
            return self.build_stimulus_matrix(config, subjects, processing_func)

    # Расстановка баллов
    def build_matrix_by_ranks(self, config, subjects, processing_func=calc_maximum, normalize_func=normalize):
        atlas_path = config['atlas_path']

        if config['save_to_numpy']:
            save_to_numpy = True
        else:
            save_to_numpy = False

        matrix = None

        # Интерируемся по объектам в конфиге
        for subject in subjects:        
            data_loader = DataLoader()  
            if save_to_numpy:
                data = data_loader.load_from_nii_and_save(
                    data_path=subject['data_path'], 
                    npy_path=subject['numpy_path'],
                    tr=subject['tr'],
                    atlas_path=atlas_path,
                    standardize=False)
            elif 'numpy_path' in subject:
                numpy_path = subject['numpy_path']
                data = data_loader.load_from_npy(numpy_path)
            else:
                data = data_loader.load_from_nii_and_save(
                    data_path=subject['data_path'], 
                    npy_path=None,
                    tr=subject['tr'],
                    atlas_path=atlas_path,
                    standardize=False)

            # Получаем и обрабатываем данные
            events = data_loader.load_events(subject['events_path'])
            if data is None or events is None:
                continue

            # Формируем объект хранящий данные
            sub = SubjectData()
            sub.set_data(data)
            sub.set_events(events)
            sub.set_tr(subject['tr'])

            runs = sub.cut_for_runs(window_size=10)
            ranks_list = list()
            for run in runs:
                processed_data = sub.apply_func(run, processing_func)
                ranks = normalize_func(processed_data)
                ranks_list.append(ranks)
            summed_ranks = np.sum(ranks_list, axis=0)

            if matrix is None:
                matrix = summed_ranks
            else:
                matrix = np.concatenate((matrix, summed_ranks), axis=0)
        # os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        # np.save(matrix_path, matrix)
        return matrix



    # Обрабатываем стимулы отдельно
    def build_stimulus_matrix(self, config, subjects, processing_func=calc_maximum):
        atlas_path = config['atlas_path']

        if config['save_to_numpy']:
            save_to_numpy = True
        else:
            save_to_numpy = False

        matrix = None

        # Интерируемся по объектам в конфиге
        for subject in subjects:        
            data_loader = DataLoader()  
            if save_to_numpy:
                data = data_loader.load_from_nii_and_save(
                    data_path=subject['data_path'], 
                    npy_path=subject['numpy_path'],
                    tr=subject['tr'],
                    atlas_path=atlas_path,
                    standardize=True)
            elif 'numpy_path' in subject:
                numpy_path = subject['numpy_path']
                data = data_loader.load_from_npy(numpy_path)
            else:
                data = data_loader.load_from_nii_and_save(
                    data_path=subject['data_path'], 
                    npy_path=None,
                    tr=subject['tr'],
                    atlas_path=atlas_path,
                    standardize=True)
                
            # Получаем и обрабатываем данные
            events = data_loader.load_events(subject['events_path'])
            if data is None or events is None:
                continue

            # Формируем объект хранящий данные
            sub = SubjectData()
            sub.set_data(data)
            sub.set_events(events)
            sub.set_tr(subject['tr'])

            runs = sub.cut_for_runs(window_size=10)
            stimulus_data = np.mean(runs, axis=0)
            
            processed_stimulus_data = sub.apply_func(stimulus_data, processing_func)

            if matrix is None:
                matrix = processed_stimulus_data
            else:
                matrix = np.concatenate((matrix, processed_stimulus_data), axis=0)
        # os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        # np.save(matrix_path, matrix)    
        return matrix



class StimulusClassifier:
    def __init__(self):
        self.logger = self._setup_logger()


    def _setup_logger(self):
        """Инициализирует и настраивает логгер для класса"""
        logger = logging.getLogger("StimulusClassifier")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
            )
            logger.addHandler(handler)
        return logger


    def fit(self, X, y, model_path=None):
        self.model = self.train_best_model_by_recall(X, y)
        if model_path is not None:
            # Сохранение модели
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
    

    def predict(self, y_test):
        return self.model.predict(y_test)

    
    def train_best_model_by_recall(self, train_matrix, y, target_class=1, random_state=42, verbose=True):
        # 1. Загрузка данных
        # matrix = np.load(train_matrix)
        X = train_matrix
        
        # Создание меток (4-й ответ в каждом блоке из 5 - ложный)
        labels = np.zeros(X.shape[0], dtype=int)
        labels[3::5] = 1  
        
        labels = y
        # Группы для валидации (по испытуемым)
        groups = np.repeat(np.arange(X.shape[0] // 5), 5)

        # 2. Пайплайн с ADASYN
        def create_pipeline(model):
            return Pipeline([
                ('scaler', StandardScaler()),
                ('adasyn', ADASYN(random_state=random_state)),  # <-- Изменено здесь
                ('feature_selector', 'passthrough'),
                ('model', model)
            ])

        # 3. Модели и параметры (без изменений)
        models = {
            "Logistic Regression": {
                'pipeline': create_pipeline(LogisticRegression(max_iter=1000)),
                'params': {
                    'feature_selector': [
                        SelectFromModel(LogisticRegression(penalty='l1', solver='saga', random_state=random_state)),
                        PCA(n_components=0.95)
                    ],
                    'model__C': [0.1, 1, 10, 100],
                    'model__class_weight': ['balanced', None],
                    'model__solver': ['lbfgs', 'saga', 'liblinear']
                }
            },
            "Random Forest": {
                'pipeline': create_pipeline(RandomForestClassifier()),
                'params': {
                    'feature_selector': [
                        SelectFromModel(RandomForestClassifier(random_state=random_state)),
                        RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=50)
                    ],
                    'model__n_estimators': [100, 200, 300],
                    'model__max_depth': [None, 10, 20],
                    'model__class_weight': ['balanced_subsample', None]
                }
            },
            "SVM": {
                'pipeline': create_pipeline(SVC(probability=True)),
                'params': {
                    'feature_selector': [PCA(n_components=0.95)],
                    'model__C': [0.1, 1, 10],
                    'model__kernel': ['linear', 'rbf'],
                    'model__class_weight': ['balanced']
                }
            },
            "XGBoost": {
                'pipeline': create_pipeline(XGBClassifier()),
                'params': {
                    'feature_selector': [SelectFromModel(XGBClassifier(random_state=random_state))],
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [3, 5, 7],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__subsample': [0.8, 1.0]
                }
            }
        }

        # 4. Обучение с групповой валидацией (без изменений)
        optimized_models = {}
        # recall_scorer = make_scorer(recall_score, pos_label=target_class)
        # balanced_acc_scorer = make_scorer(balanced_accuracy_score)
        f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=target_class)

        for name, config in models.items():
            cv_strategy = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
            gs = GridSearchCV(
                estimator=config['pipeline'],
                param_grid=config['params'],
                cv=cv_strategy,
                scoring=f2_scorer,
                n_jobs=-1,
                verbose=0
            )
            gs.fit(X, labels, groups=groups)
            optimized_models[name] = gs.best_estimator_
            
            if verbose:
                print(f"{name} - Лучшие параметры: {gs.best_params_}")
                print(f"Recall (CV): {gs.best_score_:.3f}\n")

        # 6. Создание стекинг-ансамбля (без изменений)
        ensemble = StackingClassifier(
            estimators=[(name, model) for name, model in optimized_models.items()],
            final_estimator=SVC(max_iter=1000, class_weight='balanced', probability=True),
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # 7. Оценка ансамбля с ADASYN
        logo = LeaveOneGroupOut()
        # balanced_acc_scores = []
        f2_scores = []
        
        for train_idx, val_idx in logo.split(X, labels, groups=groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            
            # Обучаем модель на оригинальных данных без балансировки
            ensemble.fit(X_train, y_train)
            y_pred = ensemble.predict(X_val)
            
            # Сохраняем метрики
            f2_scores.append(balanced_accuracy_score(y_val, y_pred))
        
        if verbose:
            print(f"Ансамбль (ADASYN) - Recall: {np.mean(f2_scores):.3f} ± {np.std(f2_scores):.3f}")
        
        return ensemble.fit(X, labels)