import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Библиотеки
import numpy as np  # Работа с матрицами и векторам
import matplotlib.pyplot as plt  # Библиотека для отрисовки
from PIL import Image
from tensorflow.keras.datasets import mnist  # Библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten


# Получаю данные из базы цифр mnist
def GetData():
    # Загрузка обучающего и тестового множества
    (x_train_, y_train), (x_test_, y_test) = mnist.load_data()

    # Стандартизация данных путем масштабирования их в диапазон от 0 до 1
    x_train_ = x_train_ / 255
    x_test_ = x_test_ / 255

    # Преобразование выходных значений в виде вектора длиною 10
    y_train_cat_ = keras.utils.to_categorical(y_train, 10)
    y_test_cat_ = keras.utils.to_categorical(y_test, 10)

    return (x_train_, y_train_cat_), (x_test_, y_test_cat_)


# Создаю модель нейроной сети, указываю количество слоев, делаю архитектуру
def CreateModel():
    # Формирование нейронной модели и вывод ее структуры в консоль
    model_ = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        # На первый слой подаю изображение 28 на 28 пикселей и 1 байтом на изображение, то есть 256 градаций серого
        Dense(128, activation='relu'),  # Скрытый слой на 128 нейронов
        Dense(10, activation='softmax')  # Выходной слой содержищий варианты из 10 цифр
    ])

    # Делаем оптимизацию c оптимизатором adam, функцией потерь categorical_crossentropy и метрикой accuracy. Они максимально подходит для классификации
    model_.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    print(model_.summary())  # Вывод структуры НС в консоль
    return model_


# Обучение модели правильному определению цифр
def TrainingModel():
    # Запуск процесса обучения 80% обучающая выборка 20% - выборка валидации(для проверки)
    model.fit(x_train, y_train_cat, batch_size=32, epochs=3, validation_split=0.2)


# Вывод 25 пяти изображений и результата их оценивания
def TestModel():
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)

    # Выделение верных вариантов
    mask = pred == np.argmax(y_test_cat, axis=1)
    # print(np.argmax(y_test_cat, axis=1))

    x_true = x_test[mask]

    # Вывод первых 25 верных результатов
    plt.figure(figsize=(10, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_true[i], cmap=plt.cm.binary)

        if i == 24:
            print(np.argmax(y_test_cat[i]))
            break

        print(np.argmax(y_test_cat[i]), end=" ")
    plt.show()


# Получение изображения которое надо обработать
def DefImage(image_path_):
    # Подаем на вход конкретное изображение
    image_path = image_path_
    img = Image.open(image_path)

    # Преобразую изображение
    size = (28, 28)
    out = img.resize(size)
    img_array = np.array(out)
    img_array_true = img_array / 255

    # Подаем на вход изображение и указываем трехмерный тенсзер со значением 0
    x = np.expand_dims(img_array_true, axis=0)

    res = model.predict(x)
    print(res)  # Выводим вектор
    print(f"Распознаная цифра {np.argmax(res)}")  # Выводим значение максимального индекса массива
    plt.imshow(img_array, cmap=plt.cm.binary)
    plt.show()


# Получаю нужные данные
(x_train, y_train_cat), (x_test, y_test_cat) = GetData()

# Создаю модель нейронной сети
model = CreateModel()

# Обучаю НС
TrainingModel()

# Тестирую НС
TestModel()

# Проверяю нейронную сеть на конкретном изображении
DefImage("/Users/ПК/Desktop/image.bmp")
