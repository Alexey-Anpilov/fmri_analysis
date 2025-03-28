import numpy as np

class SubjectData:
    def __init__(self, data=None, events=None, tr=1.0):
        self.data = data
        self.events = events
        self.tr = tr

    # Set data
    def set_data(self, data):
        self.data = data

    # Set events
    def set_events(self, events):
        self.events = events

    # Set time repetition
    def set_tr(self, tr):
        self.tr = tr

    # Get data
    def get_data(self):
        return self.data

    # Get events
    def get_events(self):
        return self.events
    
    # Get time repetition
    def get_tr(self):
        return self.tr
    

    def cut_and_apply_function(self, window_size=10, process_func=np.max, need_average=False):
        '''
            Для одного испытуемого нарезает данные, применяет фукнцию и склеивает данные, так
            что их можно подать на вход модели
        '''
        truth_data, lie_data = self.cut_answers(window_size)

        truth_data_processed, lie_data_processed = self.apply_func(truth_data, lie_data, process_func, need_average)
        
        return truth_data_processed, lie_data_processed
    

    def cut_answers(self, window_size, average=False):
        '''
            Вырезает из данных участки с импульсами согласно файлу разбиения
        '''

        # разделяем на правду и ложь
        truth_onsets = self.events[self.events['trial_type'] == 0]['onset'].values
        lie_onsets = self.events[self.events['trial_type'] == 1]['onset'].values

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
            window_volumes = int(np.round(window_size / self.tr))
            signals = []
            for onset in onsets:
                start = int(np.round(onset / self.tr))
                end = start + window_volumes
                if end > self.data.shape[0]:
                    print(f"Пропущен вопрос (выход за границы данных)")
                    continue
                window_data = self.data[start:end, :]
                if average:
                    window_data = np.mean(window_data, axis=0)  # Усреднение по времени -> [регионы]
                signals.append(window_data)
            return np.array(signals)
        
        truth_data = extract_windows(truth_onsets)
        lie_data = extract_windows(lie_onsets)
        
        return truth_data, lie_data
    

    def apply_func(self, truth_data, lie_data, process_func, need_average=False):
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
        processed_truth = process_func(truth_data)    # (25, 132)
        processed_lie = process_func(lie_data)    # (5, 132)

        if need_average:
            # 2. Усреднить максимумы по всем ответам каждого типа
            processed_truth = np.mean(processed_truth, axis=0)  # (132,)
            processed_lie = np.mean(processed_lie, axis=0)      # (132,)


        return processed_truth, processed_lie