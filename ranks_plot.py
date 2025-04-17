import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

save = True


def visualize(data, dir_name, idx):
# Создаем тепловую карту
    plt.figure(figsize=(10, 6))  # Задаем размер графика
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Значения'})  # 'viridis' — цветовая карта
    plt.title('Тепловая карта')
    plt.xlabel('Регионы')
    plt.ylabel('Стимулы')
    if save:
        plt.savefig(dir_name + f'/sub-{idx:02d}')
    else:
        plt.show()




train_matrix = './area_matrix/HC_raw_matrix_auc.npy'
test_matrix = './area_matrix/test_matrix_auc.npy'

# train_matrix, test_matrix = test_matrix, train_matrix

matrix = np.load(train_matrix)
N = matrix.shape[0]  # Длина массива
sub_num = N // 5
labels = np.zeros(N, dtype=int)  # Создаем массив из нулей
labels[3::5] = 1  # Каждый 4-й элемен
print(matrix.shape)
X = matrix
y = labels


subjects = np.array_split(matrix, sub_num)
predict_labels = np.array_split(labels, sub_num)
dir_path = './area_matrixes_visualized/test'
for idx, sub in enumerate(subjects):
    visualize(subjects[idx], dir_path, idx)

# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# def visualize_all(subjects, predict_labels, rows=6, cols=3):
#     fig, axes = plt.subplots(rows, cols, figsize=(15, 25))
#     fig.suptitle('Тепловые карты по регионам', fontsize=16)
    
#     for idx, (sub, label) in enumerate(zip(subjects, predict_labels)):
#         ax = axes[idx//cols, idx%cols]
#         sns.heatmap(sub, cmap='viridis', cbar_kws={'label': 'Значения'}, ax=ax)
#         ax.set_title(f'Субъект {idx}, Метка: {label[0]}')
#         ax.set_xlabel('Регионы')
#         ax.set_ylabel('Стимулы')
    
#     # Скрываем пустые subplots, если количество графиков не кратно rows*cols
#     for i in range(len(subjects), rows*cols):
#         axes[i//cols, i%cols].axis('off')
    
#     plt.tight_layout()
#     plt.savefig('all_heatmaps.png', dpi=300, bbox_inches='tight')
#     plt.show()

# # Загрузка данных
# train_matrix = './ranks_matrix/HC_raw_matrix_auc.npy'
# test_matrix = './ranks_matrix/test_matrix_auc.npy'
# matrix = np.load(train_matrix)
# N = matrix.shape[0]
# labels = np.zeros(N, dtype=int)
# labels[3::5] = 1

# # Разделение данных
# subjects = np.array_split(matrix, 17)
# predict_labels = np.array_split(labels, 17)

# # Визуализация всех графиков
# visualize_all(subjects, predict_labels)