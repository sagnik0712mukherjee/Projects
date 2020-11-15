from flask import Flask, request, render_template, send_file
import correlations
import model
app = Flask(__name__)

@app.route('/')
def frontend(name=None):
	return  render_template('index.html',name=name)


@app.route('/heatmap', methods=['POST'])
def heatmap():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file found!'
    fileObject = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if fileObject.filename == '':
        return 'Empty file name!'
    image_file = correlations.generateHeatmap(fileObject)
    return image_file

@app.route('/predict', methods=['POST'])
def predict():
    # check if the post request has the file part
    if 'file' not in request.files:
        return 'No file found!'
    print("printing files")
    filesList = request.files.getlist("file")
    inputsFile = filesList[0]
    targetFile = filesList[1]
    # if user does not select file, browser also
    # submit an empty part without filename
    if inputsFile.filename == '' or targetFile == '':
        return 'Empty file name!'
    result_dataframe = model.init(inputsFile, targetFile)
    return result_dataframe