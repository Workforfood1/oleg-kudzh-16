#!/bin/bash

# Установка зависимостей из requirements.txt, если файл существует
if [ -f "/requirements.txt" ]; then
    pip install -r /requirements.txt
fi

# Запуск переданного Airflow-компонента (webserver, scheduler и т.д.)
exec airflow "$@"