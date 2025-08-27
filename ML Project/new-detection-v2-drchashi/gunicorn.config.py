import multiprocessing
import os

bind = "0.0.0.0:9310"

workers = multiprocessing.cpu_count() * 2 + 1  # Recommended formula
threads = 2

timeout = 120
graceful_timeout = 30

loglevel = "info"
accesslog = os.path.join("/var/www/detection_app/logs", "gunicorn_access.log")
errorlog = os.path.join("/var/www/detection_app/logs", "gunicorn_error.log")

proc_name = "detection_app"
