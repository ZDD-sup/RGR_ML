import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


def Author():
    st.title("Разработка дашборда для моделей ML и анализа данных")

    st.header("Автор:")
    st.write("ФИО: Зайцев Дмитрий Дмитриевич")
    st.write("Группа: ФИТ-221")
    st.image("me.jpg",width =200) 


def Data():
    df = pd.read_csv('airlines_task_final.csv')

    st.title("Информация о наборе данных:")
    st.header("Тематика датасета: Задержки рейсов")
    st.header("Описание признаков:")
    st.write("- Airline - Авиакомпания. Количество используемых авиокомпаний 18 штук")
    st.write("- AirportFrom  - Аэропорт из. Количество используемых аэропортов 292 штуки")
    st.write("- AirportTo - Аэропорт в. Количество используемых аэропортов 292 штуки")
    st.write("- DayOfWeek - День недели. Количетсво используемых дней недели 7")
    st.write("- Time - Время . Значение от 10 до 1439")
    st.write("- Length - Продолжительность полёта. Значение от 0 до 655")
    st.header("Целевой параметр:")
    st.write("- Delay - Задержка рейса. Значение 0 / 1")

    st.header(" Предобработка данных:")
    st.write("Нужно предсказать задержка рейса.")
    st.write("Были удалены выбросы, дубликаты, строки с пропущеными значениями.")
    st.write("При обработке датасета была проведена нормализация данных с помощью MinMaxScaler.\
            Кодирование категориальных признаков.\
            Так же был исправлен дисбаланс бинарного целевого признака с помощью RandomOverSampler.")

    st.dataframe(data=df, width=None, height=None, use_container_width=False)


def Gharts():
    df = pd.read_csv('airlines_task_final.csv')
    df_not = pd.read_csv('airlines_task.csv')

    st.title("Датасет airlines_task")

    st.header("Тепловая карта с корреляцией между признаками")

    plt.figure(figsize=(19, 9))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)

    st.header("Гистограммы")

    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)["Airline"], bins=18)
    plt.title(f'Гистограмма для Airline')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)["AirportFrom"], bins=292)
    plt.title(f'Гистограмма для AirportFrom')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)["AirportTo"], bins=292)
    plt.title(f'Гистограмма для AirportTo')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)["DayOfWeek"], bins=7)
    plt.title(f'Гистограмма для DayOfWeek')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.histplot(df.sample(5000)["Time"], bins=100)
    plt.title(f'Гистограмма для Time')
    st.pyplot(plt)

    st.header("Ящики с усами ")

    plt.figure(figsize=(10, 6))
    sns.boxplot(df_not['Length'])
    plt.title(f'Length')
    plt.xlabel('Значение с выбрасами')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.boxplot(df['Length'])
    plt.title(f'Length')
    plt.xlabel('Значение без выбрасов')
    st.pyplot(plt)

    X = df.drop(['Delay'], axis = 1)
    y = df['Delay']
    X_resampled, y_resampled = RandomOverSampler().fit_resample(X, y)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['Delay'] = y_resampled
    st.header("Круговая диаграмма целевого признака")
    plt.figure(figsize=(8, 8))
    df_resampled['Delay'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Delay')
    plt.ylabel('')
    st.pyplot(plt)


def Model():
    st.subheader("Введите данные для предсказания:")

    input_data_reg = {}
    input_data_reg['Airline'] = st.number_input('Авиакомпания', min_value=0, max_value=17, value=12)
    input_data_reg['AirportFrom'] = st.number_input('Аэропорт из', min_value=0, max_value=291, value=252)
    input_data_reg['AirportTo'] = st.number_input('Аэропорт в', min_value=0, max_value=291, value=193)
    input_data_reg['DayOfWeek'] = st.number_input('День недели', min_value=1, max_value=7, value=5)
    input_data_reg['Time'] = st.number_input('Время вылета(Минуты)', min_value=10, max_value=1439, value=499)
    input_data_reg['Length'] = st.number_input('Продолжительость полёта(Минуты)', min_value=0, max_value=655, value=200)
    

    if st.button('Сделать предсказание'):
        df_used = pd.DataFrame([input_data_reg])

        model_catboost = load('CatBoostClassifier.joblib')
        st.success('CatBoostClassifier: 1 '+ ' Задержка рейса' if model_catboost.predict(df_used) == 1 else 'CatBoostClassifier: 0 '+'Нет задержки рейса')

        model_nb = load('gaussian_nb_model.joblib')
        st.success('GaussianNB: 1 ' + ' Задержка рейса' if model_nb.predict(df_used) == 1 else 'GaussianNB: 0 ' + 'Нет задержки рейса')

        model_bag = load('BaggingClassifier.joblib')
        st.success('BaggingClassifier: 1 ' + ' Задержка рейса' if model_bag.predict(df_used) == 1 else 'BaggingClassifier: 0 ' + 'Нет задержки рейса')

        model_clus = load('kmeans_class.joblib')
        df_used['Delay'] = 1
        data_array = df_used.values
        df_used = df_used.drop(['Delay'], axis=1)
        st.success('Kmeans: 1 ' + ' Задержка рейса' if model_clus.predict(data_array) == 1 else 'Kmeans: 0 ' +'Нет задержки рейса')

        model_stac = load('StackingClassifier.joblib')
        st.success('StackingClassifier: 1 ' + ' Задержка рейса' if model_stac.predict(df_used) == 1 else 'StackingClassifier: 0 ' + 'Нет задержки рейса')

        model_sq = load_model('Sequential.h5')
        st.success('Sequential: 1 ' + ' Задержка рейса' if (np.around(model_sq.predict(df_used))[0]) == 1 else 'Sequential: 0 ' + 'Нет задержки рейса')


st.sidebar.title('Навигация:')
page = st.sidebar.selectbox("Выберите страницу", ["Разработчик", "Датасет", "Визуализация", "Инференс модели"])

if page == "Разработчик":
    st.title('Расчётно графичесикая работа ML')
    st.header("Тема РГР:")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")
    Author()
elif page == "Датасет":
    Data()
elif page == "Визуализация":
    Gharts()
elif page == "Инференс модели":
    Model()