import os

from flask import Flask, flash, request, redirect, url_for, render_template
from src.models.predict_SignatureForgeryDetectorLib import isForgery as fD

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def performSignatureVerification (personId, compareImgPath):
    resp = fD(personId,compareImgPath)
    return "PersonId: " + str(personId) + ": " + resp

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True, static_folder=UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/')
    def indexPage():
        response = '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Uploads new File</h1>
        <form method=post enctype=multipart/form-data action="/upload">
		  <input type=text name=personId>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''
        full_filename = ""
        result = request.args.get('result')
        if request.args.get('result'):
            full_filename = request.args.get('filePath')
        #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'NFI-00101001.png')
        return render_template("index.html", filePath = full_filename, result=result)
        
        #return response;


    def allowed_file(filename):
        return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                #filename = secure_filename(file.filename)
                filename = file.filename
                
                #imgFilePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                imgFilePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(imgFilePath)
                
                result = performSignatureVerification(int(request.form.get("personId")), imgFilePath)
                #result = performSignatureVerification(49, imgFilePath)
                
                return redirect("/?result=" + result + "&filePath=" + imgFilePath ) #redirect(url_for('download_file', name=filename))
        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        '''


    return app

   
if __name__ == '__main__':
    create_app().run(debug=False, port=8000, use_reloader=False)

