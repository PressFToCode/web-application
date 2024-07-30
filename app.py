from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import plotly.io as pio
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, encoding='utf8', quoting=2, quotechar='"', delimiter='\t')

    columns_to_keep = ['datapzd', 'sumbil', 'sumpl', 'sumserv', 'sumkomsb', 'sumbilv', 'sumplv', 'sumservv', 'sumkomsbv']
    data = data[columns_to_keep]
    data['datapzd'] = pd.to_datetime(data['datapzd'])
    data['total_income'] = data[['sumbil', 'sumpl', 'sumserv', 'sumkomsb']].sum(axis=1)
    median_total_income = data['total_income'].median()
    data['total_income'] = data['total_income'].replace(0, median_total_income)

    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Обработка загруженного файла
        if 'file' not in request.files:
            return "Файл не был загружен"

        file = request.files['file']
        if file.filename == '':
            return "Файл не был выбран"

        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            return "Неверный формат файла. Пожалуйста, загрузите файл в формате .csv"

        # Загрузка и предобработка данных
        data = load_and_preprocess_data(file_path)

        # Получение дат от пользователя
        train_start_date = request.form['train_start_date']
        train_end_date = request.form['train_end_date']
        forecast_start_date = request.form['forecast_start_date']
        forecast_end_date = request.form['forecast_end_date']

        # Фильтрация данных для обучения
        data_filtered = data[(data['datapzd'] >= train_start_date) & (data['datapzd'] <= train_end_date)]
        if data_filtered.empty:
            return "Нет данных для обучения в указанный период. Пожалуйста, выберите другой период."

        features = data_filtered.drop(columns=['datapzd', 'sumbil', 'sumpl', 'sumserv', 'sumkomsb', 'total_income'])
        target = data_filtered['total_income']

        if features.empty:
            raise ValueError("Нет признаков для обучения. Проверьте отфильтрованные данные и оставленные столбцы.")

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = mae ** (1 / 2)

        # Фильтрация данных для прогноза
        future_data = data[(data['datapzd'] >= forecast_start_date) & (data['datapzd'] <= forecast_end_date)]
        if future_data.empty:
            return "Нет данных для прогнозирования на указанный период. Пожалуйста, выберите другой период."

        future_features = future_data.drop(columns=['datapzd', 'sumbil', 'sumpl', 'sumserv', 'sumkomsb', 'total_income'])
        future_predictions = model.predict(future_features)
        future_data['Predicted_total_income'] = future_predictions

        grouped_actual = future_data.groupby('datapzd')['total_income'].sum().reset_index()
        grouped_predicted = future_data.groupby('datapzd')['Predicted_total_income'].sum().reset_index()

        comparison = pd.merge(grouped_actual, grouped_predicted, on='datapzd')
        comparison.columns = ['Дата', 'Актуальный_доход', 'Прогнозируемый_доход']

        # Сортировка данных по дате
        comparison = comparison.sort_values(by='Дата').reset_index(drop=True)

        total_actual_income = comparison['Актуальный_доход'].sum()
        total_predicted_income = comparison['Прогнозируемый_доход'].sum()
        percentage_deviation = ((total_predicted_income - total_actual_income) / total_actual_income) * 100

        # Построение графика
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=comparison['Дата'], y=comparison['Актуальный_доход'],
                                 mode='lines+markers', name='Фактический доход'))
        fig.add_trace(go.Scatter(x=comparison['Дата'], y=comparison['Прогнозируемый_доход'],
                                 mode='lines+markers', name='Прогнозируемый доход'))
        fig.update_layout(title='Сравнение фактического и прогнозируемого дохода',
                          xaxis_title='Дата', yaxis_title='Доход')

        graph_html = pio.to_html(fig, full_html=False)

        comparison['Актуальный_доход'] = comparison['Актуальный_доход'].apply(lambda x: f'{x:,.2f}')
        comparison['Прогнозируемый_доход'] = comparison['Прогнозируемый_доход'].apply(lambda x: f'{x:,.2f}')

        return render_template('result.html', table=comparison.to_html(classes='table table-striped'),
                               percentage_deviation=percentage_deviation, rmse=rmse, graph_html=graph_html)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
