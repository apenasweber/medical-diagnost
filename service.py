import json
import os
from datetime import datetime

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils.helpers import Helpers

class ClinicalNERService:
    def __init__(self):
        # Carregando configurações
        with open("config.json", "r") as f:
            config = json.load(f)
        self.model_name = config["model_name"]
        self.num_labels = config["num_labels"]
        self.dropout_prob = config["dropout_prob"]
        self.id2label = config["id2label"]
        self.tokenizer_config = config["tokenizer_config"]

        # Inicializando o modelo e o tokenizador
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_config)
        self.model.eval()

    def predict(self, input_data):
        # Preparando entrada
        input_ids = self.tokenizer.encode(input_data["texto_prontuario"], return_tensors="pt")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Obtendo previsões
        outputs = self.model(input_ids, attention_mask=attention_mask)
        labels = outputs[0].argmax(-1)

        # Convertendo previsões para labels
        predicted_labels = [self.id2label[i] for i in labels[0].tolist()]

        # Detectando câncer
        cancer_detected, patient_data = Helpers.detect_cancer(input_data, predicted_labels)

        return cancer_detected, patient_data
