from config import S3_BUCKET, S3_KEY, S3_SECRET
import keras
import boto3
import tempfile


s3 = boto3.client(
    's3',
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET)

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
        gotten_model=keras.models.load_model(f"{tempdir}/{model_name}")
print(gotten_model.summary())