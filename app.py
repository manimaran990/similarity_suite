'''
	flask app to get input from user and predict the similarities
'''
import os
from os import listdir
from os.path import isfile
from flask import Flask, render_template, request, redirect, url_for,send_from_directory, jsonify
import gensim  
from werkzeug.utils import secure_filename
from sentsim_gen import *

#constants
dirname, filename = os.path.split(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(dirname, 'files')
MODEL_FOLDER = os.path.join(dirname, 'models')
#allowed extension to upload
ALLOWED_EXTENSIONS = set(['csv'])
MODEL_SELECTED = ""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER']  = MODEL_FOLDER

@app.route("/")
def home():
	list_models = [ file for file in listdir(app.config['MODEL_FOLDER']) ]	
	return render_template("home.html", models=list_models)

@app.route("/trainmodel", methods=['POST','GET'])
def trainmodel():
	if request.method == 'POST':		
		filename = os.path.join(app.config['UPLOAD_FOLDER'],request.form['filename'])
		column1 = request.form['id_col']
		column2 = request.form['text_col']
		model_name = request.form['modelname']
		model = ModelGenerate(filename, column2, column1)
		model.save_model(model_name)
	return redirect(url_for("newmodel"))


@app.route("/getsim", methods=['POST','GET'])
def getsim():
	newtext = request.form['newtext']
	try:
		model_selected = os.path.join(app.config['MODEL_FOLDER'], request.form['options'])
	except Exception as e:
		model_selected = MODEL_SELECTED

	#using model to get similare texts	
	model = ModelGenerate()
	new_sentence = model.tokenize_words(newtext.lower().split())
	model = gensim.models.Doc2Vec.load(model_selected)
	result = model.docvecs.most_similar(positive=[model.infer_vector(new_sentence)],topn=10)

	results = []
	for row in result:
		results += ["{} - {:.2f}%\n".format(row[0], float(row[1])*100) ]

	list_models = [ file for file in listdir(app.config['MODEL_FOLDER']) ]	

	return render_template("home.html", simresults=results, subtext=newtext, tokenizedtext=new_sentence, models=list_models)

@app.route("/newmodel", methods=['POST', 'GET'])
def newmodel():
	filename = ""
	if request.method == 'POST':
		file = request.files['newfile']		
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
		
	return render_template('newmodel.html', filename=filename)

if __name__ == '__main__':
	app.run(port = int(os.environ.get('PORT', 5000)))
