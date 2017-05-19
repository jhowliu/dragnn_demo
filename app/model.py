import os
import sys
import requests

sys.path.append(os.path.abspath('../'))
import model_dragnn as model

segmentation_url = "http://192.168.10.108:3013/simplesegment"

dragnn_models = None

resource_path = '/'.join([os.path.abspath('./app/static'), 'models_chinese'])
print "resource path {} ".format(resource_path)
master_spec_name = '/'.join([resource_path, 'parser_spec.textproto'])
print "master_spec_name path {} ".format(master_spec_name)
checkpoint_name = '/'.join([resource_path, 'checkpoint.model'])
print "checkpoint name path {} ".format(checkpoint_name)

# return a dicitonary
def segmentation(raw_text):
    payload = {"q" : raw_text}

    response = requests.get(segmentation_url, payload)

    return response.json()

def load_model(resource_path, master_spec_name, checkpoint_name):
    global dragnn_models
    del dragnn_models

    dragnn_models = model.load_model(master_spec_name, resource_path, checkpoint_name)

    return None

load_model(resource_path, master_spec_name, checkpoint_name)

def test_model(raw_text):
    segmentation_result = segmentation(raw_text) # return a dictionary
    tokens = segmentation_result['segmentresult']

    input_text = " ".join(tokens) # because the dragnn is split by space in default

    parsed_sentence = model.inference(dragnn_models['session'], dragnn_models['graph'],
                                      dragnn_models['builder'], dragnn_models['annotator'],
                                      input_text, False)

    result = model.parse_to_conll(parsed_sentence)

    return result


