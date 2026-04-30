# matrix-convolution-project
# Вычисление свёртки матриц — Сетевые технологии

## 🌐 Сайт
**[http://w413105.vdi.mipt.ru/](http://w413105.vdi.mipt.ru/)**

---

## 🎯 Что делает проект

Пользователь вводит две матрицы, сервер вычисляет их **свёртку** и показывает результат.


---

## 🛠 Технологии

- **Python + Flask** — серверная логика
- **Apache** — веб-сервер (прокси)
- **HTML/CSS** — интерфейс


---

## 📁 Файлы проекта

- `app.py` — основной код (Flask + свёртка)
- `requirements.txt` — зависимости (Flask)
- `README.md` — документация

---

## 🚀 Запуск на сервере

```bash
# Установка
apt install python3 python3-pip apache2 -y
pip install flask

# Запуск Flask
nohup python3 app.py &

# Настройка Apache
a2enmod proxy_http
systemctl restart apache2

