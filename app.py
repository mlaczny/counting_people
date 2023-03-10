from flask import Flask

UPLOAD_FOLDER = 'static/vid/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024