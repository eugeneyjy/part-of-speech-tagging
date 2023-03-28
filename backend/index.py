from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
import os

import torch
from nltk import word_tokenize

from models.bilstm.architecture import from_pretrained

app = Flask(__name__)
CORS(app)

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Specify the directory which saved model live
model_dir = os.getcwd() + "/models/bilstm"
model, text_field, tag_field = from_pretrained(model_dir)
model.to(dev)

@app.route('/tag', methods=['POST'])
def tag_sentence():
    body = request.get_json()
    sentence = body["sentence"]
    
    if sentence == "" or sentence == None:
        return jsonify({'tokens': [], 'tags': []})
    
    tokens = word_tokenize(sentence.lower())
    tokens_torch = torch.Tensor([[text_field.vocab.stoi[token] for token in tokens]]).long()
    model.eval()

    tags = model(tokens_torch.cuda(), [len(tokens_torch[0])])
    tags = [tag_field.vocab.itos[tag] for tag in torch.argmax(tags,2)[0]]

    response = {
        'tokens': tokens,
        'tags': tags
    }

    return jsonify(response)