import os
import random

# Укажите путь к директории, где лежат папки: texts, new_joints, new_joint_vecs
base_dir = "./dataset/HumanML3D/"  # <-- замените на нужный путь

# Лимиты для выборок (количество файлов)
limit_train_files = 2500   # например, 2500 файлов для обучения (1250 пар)
limit_test_files  = 2500   # 2500 файлов для теста (1250 пар)
limit_val_files   = 500    # 500 файлов для валидации (250 пар)

# Вычисляем лимиты в парах (целочисленное деление)
limit_train_pairs = limit_train_files // 2
limit_test_pairs  = limit_test_files  // 2
limit_val_pairs   = limit_val_files   // 2

def get_base_names(directory):
    """
    Получает базовые имена файлов (без расширений) из указанной директории,
    исключая файлы, имена которых начинаются с 'M'.
    """
    files = os.listdir(directory)
    base_names = set()
    for file in files:
        if file.startswith('.'):
            continue
        base, _ = os.path.splitext(file)
        if base.startswith('M'):
            continue
        base_names.add(base)
    return base_names

# Формируем пути к нужным папкам
folders = ["texts", "new_joints", "new_joint_vecs"]
dirs = [os.path.join(base_dir, folder) for folder in folders]

# Получаем множества базовых имён для каждой папки
base_sets = [get_base_names(d) for d in dirs]

# Находим пересечение базовых имён во всех папках
common_bases = set.intersection(*base_sets)
if not common_bases:
    print("Не найдено общих базовых имён во всех папках!")
    exit(1)

# Для каждого базового имени формируем два файла: base и 'M'+base
all_names = []
for base in sorted(common_bases):
    all_names.append(base)
    all_names.append("M" + base)

# Записываем полный список имён в all.txt
with open(os.path.join(base_dir, "all.txt"), "w") as f:
    for name in all_names:
        f.write(name + "\n")
print(f"Создан файл all.txt с {len(all_names)} записями.")

# Формируем список пар: каждая пара — (base, 'M'+base)
pairs = [(base, "M" + base) for base in sorted(common_bases)]
random.shuffle(pairs)
total_pairs = len(pairs)
print(f"Всего доступно {total_pairs} пар.")

# Определяем индексы для срезов по лимитам (если лимит больше, чем доступно, то используем максимально возможное)
train_end = min(limit_train_pairs, total_pairs)
test_end  = min(limit_train_pairs + limit_test_pairs, total_pairs)
val_end   = min(limit_train_pairs + limit_test_pairs + limit_val_pairs, total_pairs)

train_pairs = pairs[:train_end]
test_pairs  = pairs[train_end:test_end]
val_pairs   = pairs[test_end:val_end]

# Преобразуем пары в списки имён (каждая пара даёт 2 записи)
train_names = [name for pair in train_pairs for name in pair]
test_names  = [name for pair in test_pairs for name in pair]
val_names   = [name for pair in val_pairs for name in pair]

# Записываем обучающую выборку
with open(os.path.join(base_dir, "train.txt"), "w") as f:
    for name in train_names:
        f.write(name + "\n")
print(f"Создан файл train.txt с {len(train_names)} записями.")

# Записываем тестовую выборку
with open(os.path.join(base_dir, "test.txt"), "w") as f:
    for name in test_names:
        f.write(name + "\n")
print(f"Создан файл test.txt с {len(test_names)} записями.")

# Записываем валидационную выборку
with open(os.path.join(base_dir, "val.txt"), "w") as f:
    for name in val_names:
        f.write(name + "\n")
print(f"Создан файл val.txt с {len(val_names)} записями.")
