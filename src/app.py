import os
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from helper_methods.pred_methods import *
from helper_methods.segmentation import *
from helper_methods.forms import *
from keras.models import load_model
from keras import backend as K

__author__ = 'rahul'

app = Flask(__name__)
app.config.from_pyfile('config.cfg')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

db = SQLAlchemy(app)

# parameters of the image
HEIGHT = 224
WIDTH = 224
# loading the pretraiend model
model = None


class Patients(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fname = db.Column(db.String(30), nullable=False)
    lname = db.Column(db.String(30), nullable=False)
    username = db.Column(db.String(30), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    sex = db.Column(db.String(6), nullable=False)
    dob = db.Column(db.Date(), nullable=False)
    phoneno = db.Column(db.Integer, nullable=False)
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

@app.route('/')
@app.route('/index')
def index():
    #return render_template("upload_image.html")
    return render_template("index.html", title="Home")

@app.route('/login', methods=["POST", "GET"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@admin.com' and form.password.data == 'password':
            flash('You have been logged in@', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessfull, Please check username and password', 'danger')
    return render_template('login.html', title='login', form=form)

@app.route('/registerUser', methods=["POST", "GET"])
def registerUser():
    form = RegisterForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('index'))
    return render_template('register.html', title='register', form=form)

@app.route('/registration')
def navigate_registration():
    return render_template("registration.html", title="Register")

@app.route('/registration', methods=["POST"])
def register():
    fname = request.form["fname"]
    lname = request.form["fname"]
    username = request.form["username"]
    email_addr = request.form["email_addr"]
    sex = request.form["sex"]
    dob = request.form["dob"]
    phone_num = request.form["phone_num"]
    address_line1 = request.form["address_line1"]
    address_line2 = request.form["address_line2"]
    city = request.form["city"]
    state = request.form["state"]
    postal = request.form["postal"]
    country = request.form["country"]

    db.create_all()
    patient_info = Patients(fname=fname, lname=lname, username=username, email=email_addr, sex=sex, dob=dob, phoneno=phone_num,
                            address=address_line1+address_line2, city=city, state=state, zipcode=postal, country=country)
    print(patient_info)
    db.session.add(patient_info)
    db.session.commit()

    session['curr_patient'] = patient_info.id

    return render_template("upload_image.html")

@app.route('/upload_image', methods=["POST", "GET"])
def upload():
    original_image_target = os.path.join(APP_ROOT, 'images/original_images/')
    preprocessed_image_target = os.path.join(APP_ROOT, 'images/preprocessed_images/')
    segment_MA_target = os.path.join(APP_ROOT, 'images/microaneurysms/')
    exudates_target = os.path.join(APP_ROOT, 'images/exudates/')
    blood_vessels_target = os.path.join(APP_ROOT, 'images/blood_vessels/')
    haemorrhage_target = os.path.join(APP_ROOT, 'images/haemorrhage/')

    if not os.path.isdir(original_image_target):
        os.mkdir(original_image_target)

    if not os.path.isdir(preprocessed_image_target):
        os.mkdir(preprocessed_image_target)

    if not os.path.isdir(segment_MA_target):
        os.mkdir(segment_MA_target)

    if not os.path.isdir(exudates_target):
        os.mkdir(exudates_target)

    if not os.path.isdir(blood_vessels_target):
        os.mkdir(blood_vessels_target)

    if not os.path.isdir(haemorrhage_target):
        os.mkdir(haemorrhage_target)

    with graph.as_default():
        model = load_model('Trained_models/VGG19 Trained Model.h5')

    for upload in request.files.getlist("file"):
        print(upload)
        filename = upload.filename
        original_destination_file = "".join([original_image_target, filename])
        preprocessed_destination_file = "".join([preprocessed_image_target, filename])
        print(original_destination_file)
        upload.save(original_destination_file)
        # preprocessing
        preprocess_image(original_image_target, preprocessed_image_target, filename, HEIGHT, WIDTH)
        tim = cv2.imread(preprocessed_destination_file)
        tim = cv2.cvtColor(tim, cv2.COLOR_BGR2RGB)
        preds, stage = prediction(model, tim)
        print(preds)
        print(stage)

    if stage > 0:
        # Microaneuryms
        gray_blobs, microaneurysms_image = MA(original_image_target, filename)
        cv2.imwrite(segment_MA_target + filename, cv2.cvtColor(microaneurysms_image, cv2.COLOR_RGB2BGR))

        # Exudates
        ex = exudate(original_image_target, filename)
        cv2.imwrite(exudates_target + filename, ex)  # cv2.cvtColor(ex, cv2.COLOR_RGB2BGR))

        # Blood Vessels
        bv = extract_bv(original_image_target, filename)
        cv2.imwrite(blood_vessels_target + filename, bv)  # cv2.cvtColor(bv, cv2.COLOR_RGB2BGR))

        # haemorrhages
        hmag = haemorrhage(original_image_target, filename)
        cv2.imwrite(haemorrhage_target + filename, hmag)

        fundus_image_patient = FundusImage(stage=stage, imageName=filename, patient_id=session['curr_patient'])
        print(fundus_image_patient)
        db.session.add(fundus_image_patient)
        db.session.commit()

    return render_template("result.html",
                           image_name=filename,
                           stage=stage,
                           preds=preds,
                           original_path=original_image_target,
                           preprocessed_path=preprocessed_image_target,
                           microaneurysms_path=segment_MA_target,
                           exudates_path=exudates_target,
                           blood_vessels_path=blood_vessels_target,
                           haemorrhage_path=haemorrhage_target)

# @app.route('/upload/<filename>')
# def send_image(filename):
#    return send_from_directory("images", filename)

@app.route('/upload', methods=['GET'])
def send_original_image():
    path = request.args.get('path')
    filename = request.args.get('filename')
    print("Original " + path)
    print("Oiriginal " + filename)
    return send_from_directory(path, filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images/original_images/')
    original_image_target = os.path.join(APP_ROOT, 'images/original_images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names, original_path=original_image_target)


if __name__ == "__main__":
    app.debug = True
    # os.environ['PYTHONPATH'] = os.getcwd()
    app.run(host='0.0.0.0', port=5005)  # , use_reloader=False)
