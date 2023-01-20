import streamlit as st

from service import ClinicalNERService

@st.cache(allow_output_mutation=True)
def load_model():
    return ClinicalNERService()

def main():
    st.set_page_config(page_title="Clinical Named Entity Recognition", page_icon=":hospital:", layout="wide")
    st.title("Clinical Named Entities Recognition")
    st.markdown("Aplicação para reconhecimento de entidades nomeadas em textos médicos.")

    model = load_model()

    texto_prontuario = st.text_area("Insira o texto do prontuário:")
    if st.button("Enviar"):
        result = model.predict({"texto_prontuario": texto_prontuario})
        cancer_detected, patient_data = model.detect_cancer(result)
        if cancer_detected:
            st.success("Câncer detectado!")
        else:
            st.warning("Câncer não detectado.")

if __name__ == "__main__":
    main()
