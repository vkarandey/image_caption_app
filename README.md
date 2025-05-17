# Сервис генерации подписи к картинкам (Image Captioning)

Это веб-приложение на Flask, которое генерирует подписи к изображениям с помощью кастомной модели на основе BLIP + дистилляции знаний.

## Возможности

- Загрузка изображения через веб-интерфейс
- Генерация описания с помощью обученной модели
- Автоматическая загрузка весов модели с Google Drive

---

##  Установка

Нужен Python 3.8 или выше.

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Акцивация venv
macOS / Linux:
``` bash
python3 -m venv venv
source venv/bin/activate
```
Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```
### 3. Установка зависимостей
```
pip install -r requirements.txt
```

## Загрузка весов модели

При первом запуске приложение автоматически скачает веса модели `student_epoch6.pt` из Google Drive.

Если это не сработает, вы можете скачать веса вручную:

[Скачать веса модели (Google Drive)](https://drive.google.com/file/d/1w7hY_dpYc-QJ_qUzBkz-2uBqxnfS0lko/view?usp=sharing)

Скачанный файл необходимо поместить в корень проекта рядом с `app.py`.

## Запуск

```bash
python app.py
```
Откройте браузер и перейдите по адресу: http://127.0.0.1:5000

## Структура проекта
```
your-repo-name/
│
├── app.py                  # Flask-приложение
├── student_epoch6.pt       # Вес модели (автоскачивание или вручную)
├── requirements.txt        # Зависимости
├── README.md               # Документация
├── templates/
│   └── index.html          # HTML-шаблон
└── static/
    └── uploaded.jpg        # Сюда сохраняется загруженное изображение
```
