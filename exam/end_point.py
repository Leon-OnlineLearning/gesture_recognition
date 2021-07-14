import os
from flask import Flask,request,Response,jsonify
from flask_api import status
from werkzeug.utils import secure_filename
from flask import send_from_directory
import numpy as np
from exam.gesture_recognition import gesture_recognition

UPLOAD_FOLDER = './recordings'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/exams/<user_id>', methods=[ 'POST'])
def receving_chunks_during_exam(user_id):
    file = request.files['chunk'] #the video
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        datapath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(datapath)
        gesture = gesture_recognition(datapath)
        # os.remove(datapath)
        return jsonify({'gesture':gesture})

    #if there is no file or bad file recived return 400
    return "Inavalid video type", status.HTTP_400_BAD_REQUEST

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=50000, debug=True)