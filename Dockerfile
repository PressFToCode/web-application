# Использую базовый образ питона
FROM python:latest

# Метаинформация и создателе образа
LABEL name_and_mail="Arslan blin.arslan@gmail.com"

# Рабочая директория
WORKDIR /app

# Перенос содержимого проекта в образ
COPY . .

# Команда по установке зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Порт контейнера
EXPOSE 5000

# Команда запуска приложения
CMD ["python", "app.py"]
