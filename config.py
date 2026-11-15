import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'brain_tumor.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'static/uploads')
    RESULTS_FOLDER = os.path.join(basedir, 'static/results')
    REPORTS_FOLDER = os.path.join(basedir, 'static/reports')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB max upload limit