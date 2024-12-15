import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'data/2019-Dec.csv' # взял из целого датасета из нескольких csv, целый убивал мне память

graphics_folder = 'Graphics'

def analyze_memory_usage(df, output_file=None):
    memory_info = [] #словарь со значениями
    total_memory = df.memory_usage(deep=True).sum() #общий обьем\ deep - доп. обьемы текстовые
    for column in df.columns:
        column_memory = df[column].memory_usage(deep=True) #память одной
        memory_info.append({  #словарь со значениями
            'column': column,
            'memory_bytes': column_memory,
            'memory_share': column_memory / total_memory,   #одна колонка из всей памяти
            'dtype': str(df[column].dtype)
        })
    if output_file:
        with open(output_file, 'w') as json_file:
            json.dump(memory_info, json_file, indent=4)
    return total_memory, memory_info


# считываем первую 1000 строк
df_sample = pd.read_csv(file_path, nrows=1000)
initial_memory, memory_info = analyze_memory_usage(df_sample, output_file='initial_memory_analysis.json') #инфо о выбореке в 1000 строк


# для строковых
def optimize_objects(df):
    for column in df.select_dtypes(include=['object']).columns:
        if df[column].nunique() / len(df[column]) < 0.5: # если уникальных меньше 50%
            df[column] = df[column].astype('category')
    return df

# для целочисленных
def optimize_integers(df):
    for column in df.select_dtypes(include=['int']).columns:
        df[column] = pd.to_numeric(df[column], downcast='integer') #берем максимальное и минимальное в колонке, на этом основании понижаем до минимально возможного
    return df


def optimize_floats(df):
    for column in df.select_dtypes(include=['float']).columns:
        df[column] = pd.to_numeric(df[column], downcast='float') # аналогичен для целочисленных
    return df


#после оптимизации
def analyze_and_compare(file_path):
    df = pd.read_csv(file_path)
    initial_memory, _ = analyze_memory_usage(df, output_file='full_data_initial_memory.json') #инфо о всех данных

    # Оптимизация типов
    df = optimize_objects(df)
    df = optimize_integers(df)
    df = optimize_floats(df)

    # Анализ памяти после оптимизации
    optimized_memory, memory_info = analyze_memory_usage(df, output_file='optimized_memory_analysis.json')

    file_size_on_disk = os.path.getsize(file_path) / (1024 ** 2)  # размер файла в мегабайтах

    memory_comparison = {
        'file_size_on_disk_MB': file_size_on_disk, #размер файла на диске
        'initial_memory_MB': initial_memory / (1024 ** 2),  # память до оптимизации
        'optimized_memory_MB': optimized_memory / (1024 ** 2),  # память после оптимизации
        'reduction_percentage': (1 - optimized_memory / initial_memory) * 100  # Процент снижения памяти
    }

    with open('memory_comparison.json', 'w') as json_file:
        json.dump(memory_comparison, json_file, indent=4)

    return memory_comparison, memory_info


comparison, optimized_info = analyze_and_compare(file_path)


# Выборочные колонки и работа с чанками
columns_to_load = [
    'event_time', 'event_type', 'product_id', 'category_id',
    'category_code', 'brand', 'price', 'user_id', 'user_session'
]

chunk_size = 50000
selected_data = []

for chunk in pd.read_csv(file_path, usecols=columns_to_load, chunksize=chunk_size):
    chunk = optimize_objects(chunk) # оптимизируем данные для каждого чанка
    chunk = optimize_integers(chunk)
    chunk = optimize_floats(chunk)
    selected_data.append(chunk)

# объединение чанков и сохранение в файл
final_data = pd.concat(selected_data)
final_data.to_csv('optimized_subset.csv', index=False)

# Графики
df_visual = pd.read_csv('optimized_subset.csv')

# 1. Линейный график: Количество событий по дням декабря
df_visual['event_time'] = pd.to_datetime(df_visual['event_time']) #преобразуем во временные данные
df_visual.set_index('event_time', inplace=True)
events_over_time = df_visual.resample('D')['event_type'].count() #группируем по дням D и считаем события

plt.figure(figsize=(10, 6)) #ширина\высота
events_over_time.plot(kind='line')
plt.title('События по времени')
plt.ylabel('Число событий')
plt.savefig(os.path.join(graphics_folder, 'events_over_time.png'))
plt.close() #освобождаю память

# 2. Столбчатая диаграмма: Распределение типов событий
event_type_counts = df_visual['event_type'].value_counts()

plt.figure(figsize=(10, 6))
event_type_counts.plot(kind='bar', color='skyblue')
plt.title('Распределение типов событий')
plt.ylabel('Количество')
plt.savefig(os.path.join(graphics_folder, 'event_type_distribution.png'))
plt.close()

# 3. Круговая диаграмма: Бренды
brand_counts = df_visual['brand'].value_counts().head(10) #берем 10 уникальных

plt.figure(figsize=(8, 8))
brand_counts.plot(kind='pie', autopct='%1.1f%%') #процентное соотношение секторов с плавающей точкой и и знаком %
plt.title('Топ 10 брендов по встречаемости')
plt.ylabel('')
plt.savefig(os.path.join(graphics_folder, 'top_10_brands.png'))
plt.close()

# 4. Корреляция между колонками price category_id и product_id они числовые
correlation_matrix = df_visual[['price', 'category_id', 'product_id']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f') #сиборн потому что только он сработал
plt.title('Корреляционная матрица')
plt.savefig(os.path.join(graphics_folder, 'correlation_matrix.png'))
plt.close()

# 5. Гистограмма: Распределение цен
plt.figure(figsize=(10, 6))
df_visual['price'].plot(kind='hist', bins=20, color='orange')
plt.title('Распределние цен')
plt.xlabel('Цены')
plt.ylabel('Частота')
plt.savefig(os.path.join(graphics_folder, 'price_distribution.png'))
plt.close()
