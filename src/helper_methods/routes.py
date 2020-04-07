import os
import secrets
from PIL import Image
from flask import render_template, request, send_from_directory, session, redirect, url_for, flash
from helper_methods.forms import RegisterForm, LoginForm, UpdateAccountForm
from helper_methods.pred_methods import *
from helper_methods.segmentation import *
from keras.models import load_model
from helper_methods.models import Patients, FundusImage
from helper_methods import app, db, bcrypt
from flask_login import login_user, current_user, logout_user, login_required

# parameters of the image
HEIGHT = 224
WIDTH = 224
# loading the pretraiend model
model = None
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
@app.route('/index')
def index():
    #return render_template("upload_image.html")
    db.create_all()
    return render_template("index.html", title="Home")

@app.route('/login', methods=["POST", "GET"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = Patients.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful, Please check email and password', 'danger')
    return render_template('login.html', title='login', form=form)

@app.route('/registerUser', methods=["POST", "GET"])
def registerUser():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegisterForm()
    if form.validate_on_submit():
        db.create_all()
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = Patients(fname=form.fname.data, lname=form.lname.data, username=form.username.data, email=form.email.data,
                        password=hashed_password, sex=form.sex.data, dob=form.dob.data, phoneno=form.phone.data,
                        address=form.address.data, city=form.city.data, state=form.state.data, zipcode=form.zipcode.data, country=form.country.data)
        db.session.add(user)
        db.session.commit()
        session['phone'] = form.phone.data
        flash(f'Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='register', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)
    #Resize the image to 125x125 pix
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@app.route('/account', methods=["POST", "GET"])
@login_required
def account():
    profile_image_taregt = 'profile_pics/'
    if not os.path.isdir(profile_image_taregt):
        os.mkdir(profile_image_taregt)

    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.profile_image_file = picture_file
        current_user.fname = form.fname.data
        current_user.lname = form.lname.data
        current_user.username = form.username.data
        current_user.email = form.email.data
        current_user.sex = form.sex.data
        current_user.dob = form.dob.data
        current_user.phoneno = form.phone.data
        current_user.address = form.address.data
        current_user.city = form.city.data
        current_user.state = form.state.data
        current_user.zipcode = form.zipcode.data
        current_user.country = form.country.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.fname.data = current_user.fname
        form.lname.data = current_user.lname
        form.username.data = current_user.username
        form.email.data = current_user.email
        form.sex.data = current_user.sex
        form.dob.data = current_user.dob
        form.phone.data = current_user.phoneno
        form.address.data = current_user.address
        form.city.data = current_user.city
        form.state.data = current_user.state
        form.zipcode.data = current_user.zipcode
        form.country.data = current_user.country
    profile_image_file = url_for('static', filename=profile_image_taregt + current_user.profile_image_file)
    return render_template('account.html', title='Account',
                           profile_image_file=profile_image_file, form=form)

@app.route('/upload_image')
@login_required
def upload_image():
    return render_template("upload_image.html", title="Image Upload")

@app.route('/registration', methods=["POST"])
def register():
    fname = request.form["fname"]
    lname = request.form["fname"]
    username = request.form["username"]
    email_addr = request.form["email_addr"]
    password = request.form["password"]
    sex = request.form["sex"]
    dob = request.form["dob"]
    phone_num = request.form["phone_num"]
    postal_address = request.form["address_line1"]
    city = request.form["city"]
    state = request.form["state"]
    postal = request.form["postal"]
    country = request.form["country"]

    db.create_all()
    patient_info = Patients(fname=fname, lname=lname, username=username, email=email_addr, password=password, sex=sex, dob=dob, phoneno=phone_num,
                            address=postal_address, city=city, state=state, zipcode=postal, country=country)
    print(patient_info)
    db.session.add(patient_info)
    db.session.commit()

    session['curr_patient'] = patient_info.id

    return render_template("upload_image.html")

@app.route('/upload_image', methods=["POST", "GET"])
@login_required
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

        fundus_image_patient = FundusImage(stage=stage, imageName=filename, patient=current_user)
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