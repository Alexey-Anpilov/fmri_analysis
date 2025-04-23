''' 
    Скрипт для считывания данных из nii и сохранения в npy.
    Информация о пути до nii данных и о том, куда сохранять,
    берется из конфига.
'''

from fmri_processing import *


# TODO: атлас надо сделать какой нибудь глобальной переменной, которую можно будет откуда угодно достать или даже в конфиге указать просто
atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'

config_raw_hc = '/home/aaanpilov/diploma/project/configs/raw_HC_data.yaml'
config_raw_test = '/home/aaanpilov/diploma/project/configs/raw_test_data.yaml'
config_hc = '/home/aaanpilov/diploma/project/configs/HC_data.yaml'
config_test = '/home/aaanpilov/diploma/project/configs/test_data.yaml'


def load_from_nii_and_save(config_path, standartize=False):
    # Получаем даннные из конфига    
    subjects = process_config(config_path)


    # Интерируемся по объектам в конфиге
    for subject in subjects:
        # TODO: тут надо, чтобы всегда был путь до numpy, надо добавить какое-нибудь исключение.
        # TODO: можно еще передавать инфу о numpy пути, чтобы конфиги не плодить.
        # Получаем и сохраняем данные
        data_loader = DataLoader(atlas_path, standartize=standartize, tr=subject['tr'])
        data_loader.load_from_nii_and_save(subject['data_path'], subject['numpy_path'])


if __name__ == '__main__':
    #Считать сырые данные
    for config in (config_raw_test, config_raw_hc):
        load_from_nii_and_save(config, standartize=False)
    
    # Считать данные в z-score
    for config in (config_hc, config_test):
        load_from_nii_and_save(config, standartize=True)