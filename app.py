import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Classificador de Óleo Lubrificante",
    page_icon="🛢️",
    layout="centered",
)

MODEL_PATH = "best.pt"


@st.cache_resource
def carregar_modelo():
    return YOLO(MODEL_PATH)


modelo = carregar_modelo()

st.title("🛢️ Classificador de Óleo Lubrificante")
st.write(
    "Envie uma foto do óleo lubrificante e o modelo (YOLO, treinado a partir de "
    "um TCC sobre mecanização agrícola) vai classificar o estado do óleo com "
    "base na cor, seguindo a escala ASTM."
)

arquivo = st.file_uploader(
    "Selecione uma imagem do óleo (JPG ou PNG)",
    type=["jpg", "jpeg", "png"],
)

if arquivo is not None:
    imagem = Image.open(arquivo).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(imagem, caption="Imagem enviada", use_container_width=True)

    with st.spinner("Classificando..."):
        resultado = modelo.predict(imagem, verbose=False)[0]

    nomes = resultado.names
    probs = resultado.probs
    top1_idx = int(probs.top1)
    top1_classe = nomes[top1_idx]
    top1_conf = float(probs.top1conf) * 100

    with col2:
        st.subheader("Resultado")
        st.metric(label="Classificação", value=top1_classe)
        st.write(f"Confiança: **{top1_conf:.1f}%**")

        st.write("---")
        st.write("Todas as probabilidades:")
        # Ordena as classes da maior para a menor probabilidade
        indices_ordenados = probs.data.argsort(descending=True)
        for idx in indices_ordenados:
            idx = int(idx)
            classe = nomes[idx]
            valor = float(probs.data[idx]) * 100
            st.write(f"- {classe}: {valor:.1f}%")
            st.progress(min(max(valor / 100, 0.0), 1.0))
else:
    st.info("Aguardando o envio de uma imagem para classificar.")

st.write("---")
st.caption(
    "Projeto baseado em TCC sobre óleo lubrificante na mecanização agrícola, "
    "usando um modelo de classificação treinado com YOLO (Ultralytics)."
)
