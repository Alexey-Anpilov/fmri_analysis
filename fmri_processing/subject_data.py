import numpy as np
import logging

"""
Класс для хранения и обработки нейровизуализационных данных испытуемого.

- Хранит данные фМРТ, информацию о событиях и TR
- Нарезать данные на временные окна с разными условиями
- Применять аналитические функции к выделенным участкам данных
- Обрабатывать данные по отдельным прогонам эксперимента
"""
class SubjectData:
    def __init__(self, data=None, events=None, tr=1.0):
        """Инициализация объекта с данными, событиями и параметром TR"""
        self.data = data
        self.events = events
        self.tr = tr        
        self.logger = self._setup_logger()


    def _setup_logger(self):
        """Создает и настраивает логгер экземпляра класса"""
        logger = logging.getLogger("SubjectData")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
            )
            logger.addHandler(handler)
        return logger


    def set_data(self, data):
        self.data = data


    def set_events(self, events):
        self.events = events


    def set_tr(self, tr):
        self.tr = tr


    def get_data(self):
        return self.data


    def get_events(self):
        return self.events
    

    def get_tr(self):
        return self.tr


    def cut_for_truth_and_lie(self, window_size=10, process_func=np.max, need_average=False):
        """
        1. Разделение данных по окнам для правдивых и ложных ответов
        2. Применение аналитической функции

        need
        """
        truth_data = self.cut_answers_for_truth(window_size)
        lie_data = self.cut_answers_for_lie(window_size)

        truth_data_processed = self.apply_func(truth_data, process_func, need_average)
        lie_data_processed = self.apply_func(lie_data, process_func, need_average)
        
        return truth_data_processed, lie_data_processed
    
    
    def extract_windows(self, onsets, window_size):
        """
        Вырезает данные по временным меткам
        
        Параметры:
            onsets: массив временных меток начала событий
            window_size: длительность окна в секундах
        """
        window_volumes = int(np.round(window_size / self.tr))
        signals = []
        
        for onset in onsets:
            start = int(np.round(onset / self.tr))
            end = start + window_volumes
            
            if end > self.data.shape[0]:
                self.logger.warning("Пропущен вопрос (выход за границы данных)")
                continue
                
            window_data = self.data[start:end, :]
            signals.append(window_data)
            
        return np.array(signals)


    def cut_answers_for_truth(self, window_size):
        truth_onsets = self.events[self.events['trial_type'] == 0]['onset'].values
        return self.extract_windows(truth_onsets, window_size)


    def cut_answers_for_lie(self, window_size):
        lie_onsets = self.events[self.events['trial_type'] == 1]['onset'].values
        return self.extract_windows(lie_onsets, window_size)


    def cut_for_runs(self, window_size, average=False):
        """
        Обработка данных по отдельным прогонам эксперимента.
        
        Разбивает данные на сегменты по 6 событий, сортирует события внутри
        прогона по номеру стимула и извлекает соответствующие временные окна.
        """
        n_chunks = len(self.events) // 6
        runs = np.array_split(self.events.iloc[:n_chunks*6], n_chunks)
        unique_names_list = self.events['stimulus_number'].unique().tolist()
        res = []

        for run_ in runs:
            run = run_[1:]  # Пропуск первого события
            
            # Сортировка событий по порядку стимулов
            sorted_onset_values = run['onset'].iloc[
                run['stimulus_number'].apply(lambda x: unique_names_list.index(x)).argsort()
            ].values
            
            res.append(self.extract_windows(sorted_onset_values, window_size))
            
        return res

    def apply_func(self, data, process_func, need_average=False):
        """
        Применяет функции к вырезанным участкам.
        """
        processed_data = process_func(data)
        return np.mean(processed_data, axis=0) if need_average else processed_data