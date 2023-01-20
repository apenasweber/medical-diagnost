# import json
# from datetime import datetime
# from service import Service

# def test_predict_disorders():
#     service = Service()
#     # Testando a predição de transtornos com um exemplo de texto
#     text = "Paciente de 69 anos com ICC de etiologia isquêmica. Paciente com Sepse pulmonar em D8 tazocin (paciente não recebeu por 2 dias Atb)."
#     disorders = service.predict_disorders(text)
#     # Verificando se a lista de transtornos não está vazia
#     assert disorders != []
#     # Verificando se o câncer foi detectado
#     assert "cancer" in [disorder.lower() for disorder in disorders]

# def test_save_patient_data():
#     service = Service()
#     # Salvando dados de um paciente
#     patient_data = {
#         "patient_id": "123",
#         "birth_date": "2000-01-01",
#         "gender": "male",
#         "disorders": ["ICC", "Sepse pulmonar"],
#         "attendance_id": "456",
#         "attendance_date": "2022-12-31"
#     }
#     service.save_patient_data(patient_data)
#     # Lendo dados do paciente salvos
#     with open("patients_data/123.json") as f:
#         saved_data = json.load(f)
#     # Verificando se os dados salvos são iguais aos dados enviados
#     assert saved_data == patient_data

# def test_load_patient_data():
#     service = Service()
#     # Salvando dados de um paciente
    # patient_data = {
    #     "patient_id": "123",
    #     "birth_date": "2000-01-01",
    #     "gender": "male",
    #     "disorders": ["ICC", "Sepse pulmonar"],
    #     "attendance_id": "456",
    #     "attendance_date": "2022-12-31"
    # }
#     service.save_patient_data(patient_data)
#     # Lendo dados do paciente salvos
#     with open("patients_data/123.json") as f:
#         saved_data = json.load(f)
#     # Verificando se os dados salvos são iguais aos dados enviados
#     assert saved_data == patient_data
