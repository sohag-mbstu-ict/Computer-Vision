FROM python:3.10.14-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install gunicorn

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /var/www/detection_app/static \
    /var/www/detection_app/media \
    /var/www/detection_app/logs

RUN python manage.py collectstatic --noinput

EXPOSE 9310

CMD ["gunicorn", "DetectionApplication.wsgi:application", "-c", "gunicorn.config.py"]
