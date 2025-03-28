from fmri_processing import *
import os

from subjects_info import *
import pickle


def load_from_nii_and_save(data_loader):
    '''
        Сохраняет npy матрицы для фМРТ данных в отдельной директории
    '''
    for sub_id in sub_info:
        fmri_path = os.path.join(data_dir, sub_id, data_file)
        output_path = os.path.join(out_dir, f"{sub_id}.npy")
        data_loader.load_from_nii_and_save(fmri_path, output_path)


def load_from_npy(data_loader):
    '''
        Загружает фМРТ данные из npy и из файлов с разбиением в локальные переменные 
    '''
    subjects_data = dict()

    for sub_id in sub_info:
        fmri_path = os.path.join(out_dir, f"{sub_id}.npy")
        events_path = os.path.join(data_dir, sub_id, events_file)

        sub = SubjectData()
        sub.set_tr(sub_info[sub_id]['tr'])
        data_loader.load_data_for_sub(sub, fmri_path, events_path, True, True)

        subjects_data[sub_id] = sub
    
    return subjects_data 


# def process_and_build_matrix(self, process_func, output_dir):
#     subjects_results = []
#     for sub_id in self.sub_info:
#         truth_data, lie_data = self.cut_answers_for_subject(sub_id)
#         if truth_data is None:
#             continue
#         avg_truth, avg_lie = self.process_subject_(truth_data, lie_data, process_func)
#         subjects_results.append((avg_truth, avg_lie))
#     # Построение матрицы
#     matrix = self.
# (subjects_results)
#     np.save(output_dir, matrix)   # Тут надо добавить аутпут директорию
#     print("Матрица данных:")
#     print(matrix.shape)  # (2 * n_subjects, 132)




# def build_matrix_(self, subjects_results):
#     """
#     Собирает матрицу из результатов всех испытуемых.

#     Параметры:
#     - subjects_results: Список кортежей (avg_truth, avg_lie) для каждого испытуемого

#     Возвращает:
#     - matrix: Матрица формы (n_subjects * 2, 132)
#             Первая половина строк — правда, вторая — ложь.
#     """
#     n_subjects = len(subjects_results)
#     matrix = np.zeros((2 * n_subjects, 132))

#     # Заполнение матрицы
#     for i, (avg_truth, avg_lie) in enumerate(subjects_results):
#         matrix[i] = avg_truth          # Строки 0..n_subjects-1: правда
#         matrix[n_subjects + i] = avg_lie  # Строки n_subjects..2n_subjects-1: ложь

#     return matrix


if __name__ == '__main__':
    data_loader = DataLoader(atlas_path)
    subjects_data = load_from_npy(data_loader)

    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    for sub_id, sub in subjects_data.items():
        if sub_id in black_list:
            continue
        # Обрезаем и преобразуем данные
        processed_truth, processed_lie = sub.cut_and_apply_function(window_size=10 , process_func=calc_max_min)

        # Проверяем данные на модели
        predicted_truth = model.predict(processed_truth)
        predicted_lie = model.predict(processed_lie)

        # Выводим результаты
        print(sub_id)
        print_predict_results(sub.get_events(), predicted_truth, predicted_lie)



