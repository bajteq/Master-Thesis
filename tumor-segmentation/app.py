from flask import Flask, flash, render_template, request, redirect,  session
from werkzeug.utils import secure_filename
import keras
from main import *
import glob
import boto3
from config import S3_BUCKET, S3_KEY, S3_SECRET
import tempfile


UPLOAD_FOLDER = '/static/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


s3 = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET)




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = app.root_path +"/" +UPLOAD_FOLDER

app.secret_key = 'qazxsw'
# print(app.root_path +'/models/model_for_segmentation.h5')


response_data = s3.get_object(
    Bucket = S3_BUCKET,
    Key = 'model_for_segmentation.h5')

model_name='model_for_segmentation.h5'
response_data=response_data['Body']
response_data=response_data.read()
#save byte file to temp storage
with tempfile.TemporaryDirectory() as tempdir:
    with open(f"{tempdir}/{model_name}", 'wb') as my_data_file:
        my_data_file.write(response_data)
        #load byte file from temp storage into variable
        model1=keras.models.load_model(f"{tempdir}/{model_name}")


# model1 = keras.models.load_model(
#     's3://tumor-segmentation-thesis/model_for_segmentation.h5')
# model2 = keras.models.load_model(
#     app.root_path +'/models/model_for_segmentation2.h5')

 
@app.route('/')
def index():
    files = glob.glob(app.config['UPLOAD_FOLDER']+'*')
    delete_files(files)
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            session.clear()
            filename = secure_filename(file.filename)
            file_path = app.config['UPLOAD_FOLDER'] + filename
            session['filename'] = filename
            file.save(file_path)
            preproces_image = preprocess(file_path)
            Y = prep_matrix(preproces_image)
            mask1 = prediction(Y, model1, "1",app.root_path)
            blended_img1 = blend_images(file_path, mask1, "1",app.root_path)
            # mask2 = prediction(Y, model2, "2",app.root_path)
            # blended_img2 = blend_images(file_path, mask2, "2",app.root_path)
            return redirect('/show_image')


@app.route('/show_image')
def displayImage():
    return render_template('show_image.html', user_image=session.get('filename', None))


if __name__ == "__main__":
    app.run(debug=True)
