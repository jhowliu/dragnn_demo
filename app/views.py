from app import app
from flask import request, session, jsonify

from model import *

@app.route('/tokenizer/<raw_text>')
def tokenizer(raw_text):
    result = segmentation(raw_text)

    print result

    return jsonify(result.json())

@app.route('/dragnn/<raw_text>')
def dragnn(raw_text):

    result = test_model(raw_text) # json 

    return jsonify(result)
