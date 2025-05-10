''' 
    Скрипт для считывания данных из nii и сохранения в npy.
    Информация о пути до nii данных и о том, куда сохранять,
    берется из конфига.
'''

from fmri_processing import *


# TODO: атлас надо сделать какой нибудь глобальной переменной, которую можно будет откуда угодно достать или даже в конфиге указать просто
atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'

config_raw_test = '/home/aaanpilov/diploma/project/configs/raw_test_data.yaml'
config_test = '/home/aaanpilov/diploma/project/configs/test_data.yaml'

config_raw_card_hc = '/home/aaanpilov/diploma/project/configs/raw_card_hc_data.yaml'
config_card_hc = '/home/aaanpilov/diploma/project/configs/card_hc_data.yaml'

config_raw_card_test = '/home/aaanpilov/diploma/project/configs/raw_card_test_data.yaml'
config_card_test = '/home/aaanpilov/diploma/project/configs/card_test.yaml'

config_raw_schz = '/home/aaanpilov/diploma/project/configs/raw_schz_data.yaml'
config_schz = '/home/aaanpilov/diploma/project/configs/schz_data.yaml'

config_raw_hc = '/home/aaanpilov/diploma/project/configs/raw_hc_data.yaml'
config_hc = '/home/aaanpilov/diploma/project/configs/hc_data.yaml'

def load_from_nii_and_save(config_path, standardize=False):
    # Получаем даннные из конфига    
    subjects = process_config(config_path)

    # Интерируемся по объектам в конфиге
    for subject in subjects:
        # TODO: тут надо, чтобы всегда был путь до numpy, надо добавить какое-нибудь исключение.
        # TODO: можно еще передавать инфу о numpy пути, чтобы конфиги не плодить.
        # Получаем и сохраняем данные
        data_loader = DataLoader()
        data_loader.load_from_nii_and_save(
            data_path=subject['data_path'], 
            npy_path=subject['numpy_path'],
            tr=subject['tr'],
            atlas_path=atlas_path,
            standardize=standardize)


if __name__ == '__main__':
    raw_configs = [
        config_raw_test, 
        config_raw_hc, 
        config_raw_card_hc, 
        config_raw_card_test, 
        config_raw_schz
    ]

    configs = [
        config_hc, 
        config_test, 
        config_card_hc, 
        config_schz, 
        config_hc
    ]

    #Считать сырые данные
    for config in (config_raw_test, config_raw_hc, config_raw_card_hc, config_raw_card_test, config_raw_schz):
        load_from_nii_and_save(config, standardize=False)
    
    # Считать данные в z-score
    for config in (config_hc, config_test, config_card_hc, config_schz, config_hc):
        load_from_nii_and_save(config, standardize=True)