import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename


@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('Plik niewidoczny')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('Brak wybranego video')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        flash('Plik załadowany pomyślnie')
        return render_template('index.html', filename=filename)


@app.route('/display/<filename>')
def display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='vid/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)