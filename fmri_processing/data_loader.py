import os
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


class DataLoader:
    def __init__(self, atlas_path):
        self.atlas_path = atlas_path
        self.masker = NiftiLabelsMasker(labels_img=self.atlas_path, standardize=True, verbose=1)

    def load_from_nii_and_save(self, data_path, output_path):
        """Загружает данные из NIfTI, преобразует и сохраняет в numpy-файл"""
        if not os.path.isfile(data_path):
            print(f"Warning: File not found - {data_path}")
    
        try:
            # Обработка и сохранение данных
            fmri_data = self.masker.fit_transform(data_path)
            output_path = os.path.join(output_path)
            np.save(output_path, fmri_data)
        except Exception as e:
            print(f"Error processing data with path {data_path}: {str(e)}")
    

    def load_data_from_npy(self, data_path):
        """Загружает сохраненные numpy-массивы"""
        if os.path.isfile(data_path):
            return np.load(data_path)
        else:
            print(f"Warning: Numpy file not found - {data_path}")
            return None
        

    def load_events_data_from_csv(self, events_path):
        """Загружает метки событий из CSV-файлов"""    
        if os.path.isfile(events_path):
            return pd.read_csv(events_path)
        else:
            print(f"Warning: Events file not found - {events_path}")


    def process_events(self, file_name):
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


    def load_data_for_sub(self, sub, data_path, events_path, is_npy_path=False, is_csv_path=False):
        if is_npy_path:
            fmri_data = self.load_data_from_npy(data_path)
        else:    
            fmri_data = self.masker.fit_transform(data_path)
        
        if is_csv_path:
            events = self.load_events_data_from_csv(events_path)
        else:
            events = self.process_events(events_path)

        sub.set_data(fmri_data)
        sub.set_events(events)