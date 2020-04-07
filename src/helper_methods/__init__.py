from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

__author__ = 'rahul'

app = Flask(__name__, template_folder='../templates')
app.config.from_pyfile('config.cfg')
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)

from helper_methods import routes