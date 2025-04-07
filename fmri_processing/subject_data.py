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
            Для одного испытуемого нарезает данные -- разделяет на правду и ложь и применяет фукнцию
        '''
        truth_data = self.cut_answers_for_truth(window_size)
        lie_data = self.cut_answers_for_lie(window_size)

        truth_data_processed = self.apply_func(truth_data, lie_data, process_func, need_average)
        lie_data_processed = self.apply_func(lie_data, process_func, need_average)
        
        return truth_data_processed, lie_data_processed
    

    # функция для извлечения окон
    def extract_windows_(self, onsets, window_size):
        window_volumes = int(np.round(window_size / self.tr))
        signals = []
        for onset in onsets:
            start = int(np.round(onset / self.tr))
            end = start + window_volumes
            if end > self.data.shape[0]:
                print(f"Пропущен вопрос (выход за границы данных)")
                continue
            window_data = self.data[start:end, :]
            # if average:
            #     window_data = np.mean(window_data, axis=0)  # Усреднение по времени -> [регионы]
            signals.append(window_data)
        return np.array(signals)

    
    def cut_answers_for_truth(self, window_size):
        truth_onsets = self.events[self.events['trial_type'] == 0]['onset'].values
        return self.cut_answers(truth_onsets, window_size)


    def cut_answers_for_lie(self, window_size):
        lie_onsets = self.events[self.events['trial_type'] == 1]['onset'].values
        return self.cut_answers(lie_onsets, window_size)


    def cut_answers(self, onsets, window_size, average=False):
        '''
            Вырезает из данных участки с импульсами согласно файлу разбиения
        '''
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
        return self.extract_windows_(onsets, window_size)
    

    def cut_for_runs(self, window_size, average=False):
        runs = np.array_split(self.events, 5)
        unique_names_list = self.events['name'].unique().tolist()
        # print(unique_names_list)

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
        
        res = list()
        for run_ in runs:
            run = run_[1:]
            sorted_onset_values = run['onset'].iloc[
    run['name'].apply(lambda x: unique_names_list.index(x)).argsort()].values
            res.append(extract_windows(sorted_onset_values))
        # res = dict()
        # for name in unique_names_list:
        #     onsets = self.events[self.events['name'] == name]['onset'].values
        #     # print(name)
        #     # print(onsets)
        #     res[name] = extract_windows(onsets)
        return res

    # TODO тут нужно принимать просто данные, а не truth и lie 
    def apply_func(self, data, process_func, need_average=False):
        """
        Обрабатывает данные одного испытуемого: применяет функцию 
        для правды и лжи в каждом регионе.
        
        need_average -- усредняя по всем вопросам правды и лжи 
        """
        processed_data = process_func(data)    # (25, 132)

        if need_average:
            processed_data = np.mean(processed_data, axis=0)  # (132,)

        return processed_data