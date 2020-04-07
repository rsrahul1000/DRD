from helper_methods import db
from datetime import datetime

class Patients(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(30), nullable=False)
    lname = db.Column(db.String(30), nullable=False)
    username = db.Column(db.String(30), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(50), nullable=False)
    sex = db.Column(db.String(6), nullable=False)
    dob = db.Column(db.Date(), nullable=False)
    phoneno = db.Column(db.String(16), nullable=False)
    address = db.Column(db.Text, nullable=False)
    city = db.Column(db.String(20), nullable=False)
    state = db.Column(db.String(30), nullable=False)
    zipcode = db.Column(db.Integer, nullable=False)
    country = db.Column(db.String(30), nullable=False)

    fundus = db.relationship('FundusImage', backref='patient', lazy='dynamic')

    def __repr__(self):
        return f"Patient('{self.fname}','{self.lname}', '{self.email}')"

class FundusImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stage = db.Column(db.Integer, nullable=False)
    imageName = db.Column(db.String(50), nullable=False)
    date_added = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False) #Foriegn key to the patients table

    def __repr__(self):
        return f"FundusImage('{self.stage}','{self.imageName}', '{self.date_added}')"