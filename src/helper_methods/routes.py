import os
import secrets
from PIL import Image
from flask import render_template, request, send_from_directory, session, redirect, url_for, flash, abort
from helper_methods.forms import RegisterForm, LoginForm, UpdateAccountForm, DiagnoseForm, RequestResetForm, ResetPasswordForm
from helper_methods.pred_methods import *
from helper_methods.segmentation import *
from keras.models import load_model
from helper_methods.models import Patients, FundusImage
from helper_methods import app, db, bcrypt, mail
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message

# parameters of the image
from scipy.sparse import dia_matrix

HEIGHT = 224
WIDTH = 224
# loading the pretraiend model
model = None
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
original_image_target = os.path.join(APP_ROOT, 'images/original_images/')
preprocessed_image_target = os.path.join(APP_ROOT, 'images/preprocessed_images/')
segment_MA_target = os.path.join(APP_ROOT, 'images/microaneurysms/')
exudates_target = os.path.join(APP_ROOT, 'images/exudates/')
blood_vessels_target = os.path.join(APP_ROOT, 'images/blood_vessels/')
haemorrhage_target = os.path.join(APP_ROOT, 'images/haemorrhage/')

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

def save_profile_picture(form_picture, image_path):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(image_path, picture_fn)
    #Resize the image to 125x125 pix
    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@app.route('/account', methods=["POST", "GET"])
@login_required
def account():
    profile_image_target = 'profile_pics/'
    saving_path_profile_target = os.path.join(app.root_path, 'static/profile_pics')
    if not os.path.isdir(profile_image_target):
        os.mkdir(profile_image_target)

    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_profile_picture(form.picture.data, saving_path_profile_target)
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
    profile_image_file = url_for('static', filename=profile_image_target + current_user.profile_image_file)
    return render_template('account.html', title='Account',
                           profile_image_file=profile_image_file, form=form)

def save_oroginal_picture(form_picture, image_path):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(image_path, picture_fn)
    form_picture.save(picture_path)

    return picture_fn

@app.route('/diagnose/new', methods=["POST", "GET"])
@login_required
def new_diagnose():

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

    form = DiagnoseForm()
    if form.validate_on_submit():
        flash('Your Diagnose has been done!', 'success')
        if form.picture.data:
            filename = save_oroginal_picture(form.picture.data, original_image_target)
            preprocessed_destination_file = "".join([preprocessed_image_target, filename])
            print(filename)
            print(preprocessed_destination_file)

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

                fundus_image_patient = FundusImage(stage=stage, imageName=filename,side=form.side.data , patient=current_user)
                print(fundus_image_patient)
                db.session.add(fundus_image_patient)
                db.session.commit()

            return render_template("result.html",
                                   image_name=filename,
                                   stage=stage,
                                   original_path=original_image_target,
                                   preprocessed_path=preprocessed_image_target,
                                   microaneurysms_path=segment_MA_target,
                                   exudates_path=exudates_target,
                                   blood_vessels_path=blood_vessels_target,
                                   haemorrhage_path=haemorrhage_target)

        return redirect(url_for('result'))
    return render_template('create_diagnose.html', title='New Diagnose',
                           form=form, legend='New Diagnose')

@app.route('/diagnose')
@login_required
def existing_diagnose():
    original_image_target = os.path.join(APP_ROOT, 'images/original_images/')
    page = request.args.get('page', 1, type=int)
    diagnosis = FundusImage.query.filter_by(patient=current_user).order_by(FundusImage.date_added.desc()).paginate(page=page, per_page=5)
    return render_template('existing_diagnose.html', diagnosis=diagnosis,
                           original_path=original_image_target)

@app.route('/diagnose/<int:diagnose_id>')
@login_required
def diagnose(diagnose_id):
    diag = FundusImage.query.get_or_404(diagnose_id)
    return render_template("result.html",
                           image_name=diag.imageName,
                           stage=diag.stage,
                           original_path=original_image_target,
                           preprocessed_path=preprocessed_image_target,
                           microaneurysms_path=segment_MA_target,
                           exudates_path=exudates_target,
                           blood_vessels_path=blood_vessels_target,
                           haemorrhage_path=haemorrhage_target,
                           diagnose=diag,
                           title='result')

@app.route("/diagnose/<int:diagnose_id>/update", methods=["POST", "GET"])
@login_required
def update_diagnose(diagnose_id):
    diag = FundusImage.query.get_or_404(diagnose_id)
    if diag.patient != current_user:
        abort(403)
    form = DiagnoseForm()
    if form.validate_on_submit():
        diag.side = form.side.data
        db.session.commit()
        flash('You Diagnosis has been updated!', 'success')
        return redirect(url_for('existing_diagnose'))
    elif request.method == 'GET':
        form.side.data = diag.side
    return render_template('create_diagnose.html', title='Update Diagnose',
                           form=form, legend='Update Diagnose')

@app.route("/diagnose/<int:diagnose_id>/delete", methods=["POST"])
@login_required
def delete_diagnose(diagnose_id):
    diag = FundusImage.query.get_or_404(diagnose_id)
    if diag.patient != current_user:
        abort(403)
    print(diag.imageName)
    #delete all the images except the original one
    os.remove(preprocessed_image_target+diag.imageName)
    os.remove(segment_MA_target+diag.imageName)
    os.remove(exudates_target+diag.imageName)
    os.remove(blood_vessels_target+diag.imageName)
    os.remove(haemorrhage_target+diag.imageName)
    #delete from the database
    db.session.delete(diag)
    db.session.commit()
    flash('You Diagnosis has been deleted!', 'success')
    return redirect(url_for('existing_diagnose'))

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request',
                  sender='noreply@demo.com',
                  recipients=[user.email])
    msg.body = f'''To reset your password, visit the following link:
{url_for('reset_token', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    mail.send(msg)

@app.route('/reset_password', methods=["POST", "GET"])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user =Patients.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('An email has been sent with instructions to reset your password', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', title="Reset Password", form=form)

@app.route('/reset_password/<token>', methods=["POST", "GET"])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    user = Patients.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        db.create_all()
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        session['phone'] = form.phone.data
        flash(f'Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', title="Reset Password", form=form)


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