import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

__author__ = 'rahul'

app = Flask(__name__, template_folder='../templates')
app.config.from_pyfile('config.cfg')

db = SQLAlchemy(app)

from helper_methods import routes