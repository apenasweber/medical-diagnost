import json
import os
from datetime import datetime

class Helpers:
    @staticmethod
    def save_patient_data(patient_data, patient_id):
        # Salvando dados do paciente em um arquivo json
        with open(f"patients_data/{patient_id}.json", "w") as f:
            json.dump(patient_data, f)
    
    @staticmethod
    def load_patient_data(patient_id):
        # Carregando dados do paciente a partir do arquivo json
        with open(f"patients_data/{patient_id}.json", "r") as f:
            patient_data = json.load(f)
        return patient_data
    
    @staticmethod
    def update_patient_data(patient_data, patient_id):
        # Atualizando dados do paciente no arquivo json
        with open(f"patients_data/{patient_id}.json", "w") as f:
            json.dump(patient_data, f)
    
    @staticmethod
    def detect_cancer(patient_data, disorder_list):
        # Verificando se o câncer foi detectado
        cancer_detected = False
        for disorder in disorder_list:
            if "I-Disorder" in disorder:
                cancer_detected = True
                patient_data["cancer_detected"] = True
                patient_data["cancer_detection_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                break
        return cancer_detected, patient_data
