from helper_methods import app

if __name__ == "__main__":
    app.debug = True
    # os.environ['PYTHONPATH'] = os.getcwd()
    app.run(host='0.0.0.0', port=5005)  # , use_reloader=False) , debug=False
