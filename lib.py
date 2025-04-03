import os
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

sub_info_hc = {
'sub-01' : {'slice_num' : 546, 'tr' : 1.0},
'sub-02' : {'slice_num' : 553, 'tr' : 1.0},
'sub-03' : {'slice_num' : 457, 'tr' : 1.0},
'sub-04' : {'slice_num' : 492, 'tr' : 1.0},
'sub-05' : {'slice_num' : 929, 'tr' : 1.0},
'sub-06' : {'slice_num' : 478, 'tr' : 1.0},
'sub-07' : {'slice_num' : 546, 'tr' : 1.0},
'sub-08' : {'slice_num' : 468, 'tr' : 1.11},
'sub-09' : {'slice_num' : 433, 'tr' : 1.11},
'sub-10' : {'slice_num' : 462, 'tr' : 1.11},
'sub-11' : {'slice_num' : 441, 'tr' : 1.11},
'sub-12' : {'slice_num' : 451, 'tr' : 1.11},
'sub-13' : {'slice_num' : 416, 'tr' : 1.11},
'sub-14' : {'slice_num' : 419, 'tr' : 1.11},
'sub-15' : {'slice_num' : 420, 'tr' : 1.11},
'sub-16' : {'slice_num' : 453, 'tr' : 1.11},
'sub-17' : {'slice_num' : 404, 'tr' : 1.11},
'sub-18' : {'slice_num' : 471, 'tr' : 1.11}
}


sub_info_schz = {
'sub-01' : { 'slice_num' : 423, 'tr' : 1.11},
'sub-02' : { 'slice_num' : 524, 'tr' : 1.0},
'sub-03' : { 'slice_num' : 618, 'tr' : 1.0},
'sub-04' : { 'slice_num' : 502, 'tr' : 1.0},
'sub-05' : { 'slice_num' : 485, 'tr' : 1.0},
'sub-06' : { 'slice_num' : 531, 'tr' : 1.0},
'sub-07' : { 'slice_num' : 551, 'tr' : 1.0},
'sub-08' : { 'slice_num' : 491, 'tr' : 1.0},
'sub-09' : { 'slice_num' : 744, 'tr' : 1.1},
'sub-10' : { 'slice_num' : 399, 'tr' : 1.1},
'sub-11' : { 'slice_num' : 513, 'tr' : 1.1},
'sub-12' : { 'slice_num' : 472, 'tr' : 1.1},
'sub-13' : { 'slice_num' : 409, 'tr' : 1.1},
'sub-14' : { 'slice_num' : 426, 'tr' : 1.1},
'sub-15' : { 'slice_num' : 472, 'tr' : 1.0},
'sub-16' : { 'slice_num' : 431, 'tr' : 1.11},
'sub-17' : { 'slice_num' : 441, 'tr' : 1.11},
'sub-18' : { 'slice_num' : 495, 'tr' : 1.11},
'sub-19' : { 'slice_num' : 449, 'tr' : 1.11},
}


black_list_hc = ['sub-01', 'sub-07']
black_list_schz = ['sub-01', 'sub-17', 'sub-16', 'sub-15']



class DataLoader:
    def __init__(self, data_dir, data_file, events_file, sub_num, atlas_path, 
                 need_load_data=False, out_dir='./numpy_data', is_hc = True):
        self.data = dict()
        self.events_data = dict()

        self.data_dir = data_dir
        self.data_file = data_file
        self.sub_num = sub_num
        self.events_file = events_file
        self.atlas_path = atlas_path
        self.out_dir = out_dir

        if is_hc:
            self.sub_info = sub_info_hc
            self.black_list = black_list_hc
        else:
            self.sub_info = sub_info_schz
            self.black_list = black_list_schz

        # Создаем директорию для сохранения данных при инициализации
        os.makedirs(self.out_dir, exist_ok=True)

        if need_load_data:
            self._load_from_nii_and_save()
        
        self._load_saved_data()
        self._load_events_data()

        if len(self.data) != len(self.events_data):
            print(f"Warning: Data size != Events size")


    def _load_from_nii_and_save(self):
        """Загружает данные из NIfTI, преобразует и сохраняет в numpy-файлы"""
        
        masker = NiftiLabelsMasker(labels_img=self.atlas_path, standardize=True, verbose=1)
        
        for sub_id in self.sub_info:
            fmri_path = os.path.join(self.data_dir, sub_id, self.data_file)

            if not os.path.isfile(fmri_path):
                print(f"Warning: File not found - {fmri_path}")
                continue
        
            try:
                # Обработка и сохранение данных
                fmri_data = masker.fit_transform(fmri_path)
                output_path = os.path.join(self.out_dir, f"{sub_id}.npy")
                np.save(output_path, fmri_data)
            except Exception as e:
                print(f"Error processing {sub_id}: {str(e)}")
        
        #sub_id = f"sub-{i:02d}"


    def _load_saved_data(self):
        """Загружает сохраненные numpy-массивы"""

        for sub_id in self.sub_info:
            numpy_path = os.path.join(self.out_dir, sub_id + '.npy')

            if os.path.isfile(numpy_path):
                self.data[sub_id] = np.load(numpy_path)
            else:
                print(f"Warning: Numpy file not found - {numpy_path}")
        

    def _load_events_data(self):
        """Загружает метки событий из CSV-файлов"""
        for sub_id in self.sub_info:
            events_path = os.path.join(self.data_dir, sub_id, self.events_file)
            
            if os.path.isfile(events_path):
                try:
                    self.events_data[sub_id] = pd.read_csv(events_path)
                except Exception as e:
                    print(f"Error loading events for {sub_id}: {str(e)}")
            else:
                print(f"Warning: Events file not found - {events_path}")


    def get_data(self):
        """Возвращает копию загруженных данных"""
        return self.data.copy()


    def get_events_data(self):
        """Возвращает копию меток событий"""
        return self.events_data.copy()
    

class DataProcessor:
    def __init__(self, data, events, tr=1.0, window_size=10, is_hc=True):
        self.data = data
        self.events = events
        self.tr = tr
        self.window_size = window_size
        
        if is_hc:
            self.sub_info = sub_info_hc
            self.black_list = black_list_hc
        else:
            self.sub_info = sub_info_schz
            self.black_list = black_list_schz


    def cut_answers(self, data, events, tr, average=False):
        # разделяем на правду и ложь
        truth_onsets = events[events['trial_type'] == 0]['onset'].values
        lie_onsets = events[events['trial_type'] == 1]['onset'].values

        # # Получаем truth_onsets (onset для trial_type == 0)
        # truth_onsets = [
        #     events['onset'][i]  # Берем onset
        #     for i in range(len(events['trial_type']))  # Итерируем по индексам
        #     if events['trial_type'][i] == 0  # Условие: trial_type == 0
        # ]

        # # Получаем lie_onsets (onset для trial_type == 1)
        # lie_onsets = [
        #     events['onset'][i]  # Берем onset
        #     for i in range(len(events['trial_type']))  # Итерируем по индексам
        #     if events['trial_type'][i] == 1  # Условие: trial_type == 1
        # ]

        # функция для извлечения окон
        def extract_windows(onsets):
            window_volumes = int(np.round(self.window_size / tr))
            signals = []
            for onset in onsets:
                start = int(np.round(onset / tr))
                end = start + window_volumes
                if end > data.shape[0]:
                    print(f"Пропущен вопрос (выход за границы данных)")
                    continue
                window_data = data[start:end, :]
                if average:
                    window_data = np.mean(window_data, axis=0)  # Усреднение по времени -> [регионы]
                signals.append(window_data)
            return np.array(signals)
        
        truth_data = extract_windows(truth_onsets)
        lie_data = extract_windows(lie_onsets)
        
        return truth_data, lie_data


    def cut_answers_for_subject(self, sub_id, average=False):
        '''
            average -- усредняет сигнал в рамках региона
        '''

        if sub_id not in self.sub_info:
            raise ValueError('Error: Unknown sub: {sub_id}')            
        
        if sub_id in self.black_list:
            return None, None
            raise ValueError(f'Warning: some problems with data for {sub_id}')

        data = self.data[sub_id]
        events = self.events[sub_id]
        tr = self.sub_info[sub_id]['tr']

        print(sub_id)

        return self.cut_answers(data, events, tr, average)
    

    def process_and_build_matrix(self, process_func, output_dir):
        subjects_results = []
        for sub_id in self.sub_info:
            truth_data, lie_data = self.cut_answers_for_subject(sub_id)
            if truth_data is None:
                continue
            avg_truth, avg_lie = self.process_subject_(truth_data, lie_data, process_func)
            subjects_results.append((avg_truth, avg_lie))
        # Построение матрицы
        matrix = self.build_matrix_(subjects_results)
        np.save(output_dir, matrix)   # Тут надо добавить аутпут директорию
        print("Матрица данных:")
        print(matrix.shape)  # (2 * n_subjects, 132)


    def build_matrix_(self, subjects_results):
        """
        Собирает матрицу из результатов всех испытуемых.

        Параметры:
        - subjects_results: Список кортежей (avg_truth, avg_lie) для каждого испытуемого

        Возвращает:
        - matrix: Матрица формы (n_subjects * 2, 132)
                Первая половина строк — правда, вторая — ложь.
        """
        n_subjects = len(subjects_results)
        matrix = np.zeros((2 * n_subjects, 132))

        # Заполнение матрицы
        for i, (avg_truth, avg_lie) in enumerate(subjects_results):
            matrix[i] = avg_truth          # Строки 0..n_subjects-1: правда
            matrix[n_subjects + i] = avg_lie  # Строки n_subjects..2n_subjects-1: ложь

        return matrix


    def process_subject_(self, truth_data, lie_data, process_func=np.max):
        """
        Обрабатывает данные одного испытуемого: вычисляет усредненные максимальные значения
        для правды и лжи в каждом регионе.

        Параметры:
        - truth_data: Массив формы (n_truth_trials, n_timepoints, n_regions)
        - lie_data: Массив формы (n_lie_trials, n_timepoints, n_regions)

        Возвращает:
        - avg_truth: Усредненные максимумы для правды (132 региона)
        - avg_lie: Усредненные максимумы для лжи (132 региона)
        """
        # 1. Найти максимум по времени для каждого ответа и региона
        max_truth = process_func(truth_data)    # (25, 132)
        max_lie = process_func(lie_data)    # (5, 132)

        # 2. Усреднить максимумы по всем ответам каждого типа
        avg_truth = np.mean(max_truth, axis=0)  # (132,)
        avg_lie = np.mean(max_lie, axis=0)      # (132,)

        return avg_truth, avg_lie

    # def save_and_plot_responses(self, sub_id, region_id, output_dir="results"):
    #     """
    #     Сохраняет графики для всех ответов (правда и ложь) в отдельные файлы,
    #     а также выводит и сохраняет все графики вместе.

    #     Параметры:
    #     - sub_id: Идентификатор испытуемого
    #     - region_id: Индекс региона для визуализации
    #     - output_dir: Папка для сохранения графиков
    #     """

    #     truth_data, lie_data = self.cut_answers_for_subject(sub_id)
        
    #     # Проверка и выбор региона
    #     n_regions = truth_data.shape[2]
    #     if region_id >= n_regions:
    #         raise ValueError(f"Атлас содержит только {n_regions} регионов. Индекс {region_id} недопустим.")
    #     output_dir += f'/sub-{sub_id}_region{region_id}'
    #     # Создание папки для сохранения графиков
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     # Временная ось
    #     time_points = np.arange(0, self.window_size, self.tr)
        
    #     # Сохранение отдельных графиков для правды
    #     truth_plots = []
    #     for i in range(len(truth_data)):
    #         plt.figure(figsize=(6, 4))
    #         plt.plot(time_points, truth_data[i, :, region_id], color='green', label='Правда')
    #         plt.xlabel("Время (сек)")
    #         plt.ylabel("Сигнал")
    #         plt.grid(alpha=0.3)
            
    #         # Сохранение в файл
    #         plot_path = os.path.join(output_dir, f"{sub_id}_region_{region_id}_truth_{i+1}.png")
    #         plt.savefig(plot_path, bbox_inches='tight')
    #         plt.close()
            
    #         # Сохранение для объединённого графика
    #         truth_plots.append(plot_path)
        
    #     # Сохранение отдельных графиков для лжи
    #     lie_plots = []
    #     for i in range(len(lie_data)):
    #         plt.figure(figsize=(6, 4))
    #         plt.plot(time_points, lie_data[i, :, region_id], color='red', label='Ложь')
    #         plt.xlabel("Время (сек)")
    #         plt.ylabel("Сигнал")
    #         plt.grid(alpha=0.3)
            
    #         # Сохранение в файл
    #         plot_path = os.path.join(output_dir, f"{sub_id}_region_{region_id}_lie_{i+1}.png")
    #         plt.savefig(plot_path, bbox_inches='tight')
    #         plt.close()
            
    #         # Сохранение для объединённого графика
    #         lie_plots.append(plot_path)
        
    #     # Создание объединённого графика
    #     fig, axes = plt.subplots(6, 5, figsize=(25, 20))
    #     fig.suptitle(f"Регион {region_id} | Испытуемый {sub_id}\n25 ответов-правды и 5 ответов-лжи", 
    #                 y=1.02, fontsize=14)
        
    #     # Визуализация правдивых ответов (25 графиков)
    #     for i, plot_path in enumerate(truth_plots):
    #         ax = axes[i // 5, i % 5]
    #         img = plt.imread(plot_path)
    #         ax.imshow(img)
    #         ax.axis('off')  # Отключение осей
    #         ax.set_title(f"Правда #{i+1}", fontsize=8)
        
    #     # Визуализация ложных ответов (5 графиков)
    #     for i, plot_path in enumerate(lie_plots):
    #         ax = axes[5, i]
    #         img = plt.imread(plot_path)
    #         ax.imshow(img)
    #         ax.axis('off')  # Отключение осей
    #         ax.set_title(f"Ложь #{i+1}", fontsize=8)
        
    #     # Сохранение объединённого графика
    #     combined_path = os.path.join(output_dir, f"{sub_id}_region_{region_id}_combined.png")
    #     plt.savefig(combined_path, bbox_inches='tight')
    #     plt.close()
        
    #     # Вывод объединённого графика
    #     plt.figure(figsize=(25, 20))
    #     img = plt.imread(combined_path)
    #     plt.imshow(img)
    #     plt.axis('off')
    #     plt.show()