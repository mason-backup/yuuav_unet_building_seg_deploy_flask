# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
import shutil

from predict import demo_api

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd() + "/images"
app.config['RESULT_FOLDER'] = os.getcwd() + "/results"
# app.config['MAX_CONTENT_LENGTH'] = 8 * 256 * 256


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/result/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'],
                               filename)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/uploadImg', methods=['POST'])
def upload_file():
    shutil.rmtree('./images')
    os.mkdir('images')
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_url = url_for('uploaded_file', filename=filename)

        return jsonify(name=filename, filename=file_url)


@app.route('/predictImg', methods=['POST'])
def predict_img():
    # os.system('sh demo.sh')
    pct = demo_api.main()
    file_list = os.listdir('./results')
    for file_ in file_list:
        fmat = file_.split('.')[-1]
        if str(fmat) == 'png' or str(fmat) == 'jpg':
            result_name = file_
            result_url = url_for('result_file', filename=result_name)
            return jsonify(name=result_name, filename=result_url, area=pct)


if __name__ == '__main__':
    app.run()
