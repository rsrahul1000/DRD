# Diabetic Retinopathy Detection (DRD)
>- DRD website implemented using flask
>- This model predicts the fundus image into 5 stages
>>- Stage 1: No DR
>>- Stage 2: Mild Non-Proliferative DR
>>- Stage 3: Moderate Non-Proliferative DR
>>- Stage 4: Severe Non-Proliferative DR
>>- Stage 5: Proliferative DR

### Environment Setup
>- create and activate environment in anaconda distribution using
> ```bash
> conda create --name drdenv python=3.6 anaconda
> conda activate
> ```
>- install the dependent packages
> ```bash
> conda install -c anaconda opencv
> conda install -c anaconda keras-gpu
> conda install -c anaconda flask -y
> conda install -c anaconda flask-sqlalchemy -y
> conda install -c anaconda flask-wtf -y
> conda install -c anaconda flask-login -y
> conda install -c anaconda flask-bcrypt -y
> conda install -c anaconda werkzeug=0.16.1
> conda install -c conda-forge phonenumbers
> conda install -c conda-forge flask-mail
> pip install mysqlclient
> ```
>- in absence of GPU, install keras instead of keras-gpu.

### Trained model
>- To download the Trainined Model, [click here](https://drive.google.com/file/d/166J0WdyKn2eTDj3ZfYGnXhQqgE34NgET/view?usp=sharing)
>- Make sure the Trained Model is in 'src\Trained_Models\\'

### Application Running
>- clone [this](https://github.com/rsrahul1000/DRD.git) repository and move it in DRD project of PyCharm
>- run run.py to start the application
>- the application can be accessed using the local system IPv4_address:5005  
>- IPv4 address of system can be found using the following command in command prompt
> ```bash
> $ ipconfig
> ```
>- Example of accessing the website in browser:
> ```bash
> 192.168.X.X:5005
> ```

### Application Tested On:
>- IDE: Jetbrains Pycharm
>- GPU: Nvidia Geforce 940mx
>- OS: Windows 10