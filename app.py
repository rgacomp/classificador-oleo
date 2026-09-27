import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Classificador de Óleo Lubrificante",
    page_icon="🚜",
    layout="centered",
)

MODEL_PATH = "best.pt"

# ---------- Estilo ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Zilla+Slab:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #F7F1E3 0%, #F1E8D3 100%);
    }

    .header-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
        padding-bottom: 10px;
        border-bottom: 3px solid #B34700;
        margin-bottom: 6px;
    }
    .header-wrap .icon {
        font-size: 2.6rem;
        line-height: 1;
    }
    .header-wrap h1 {
        margin: 0;
        color: #2E2013;
        font-family: 'Zilla Slab', serif;
        font-size: 2.1rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #5C6B3B;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 6px 0 22px 0;
    }
    .intro-text {
        color: #4A3B2A;
        font-size: 1rem;
        margin-bottom: 22px;
    }

    .result-card {
        border-radius: 10px;
        padding: 20px 22px;
        color: #FFF7EC;
        font-family: 'Zilla Slab', serif;
        margin-bottom: 14px;
    }
    .result-card .classe {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        text-transform: capitalize;
    }
    .result-card .conf {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        opacity: 0.92;
    }

    .prob-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #4A3B2A;
        margin-top: 10px;
        margin-bottom: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_modelo():
    return YOLO(MODEL_PATH)


def cor_da_classe(nome: str) -> str:
    n = nome.lower()
    if "novo" in n:
        return "#5C6B3B"  # verde campo
    if "interm" in n:
        return "#B8860B"  # âmbar
    if "velho" in n or "usad" in n:
        return "#7A3A12"  # ferrugem
    return "#4A3B2A"


modelo = carregar_modelo()

st.markdown(
    '<div class="header-wrap"><span class="icon">🚜</span>'
    "<h1>Classificador de Óleo Lubrificante</h1></div>"
    '<div class="subtitle">Diagnóstico visual para mecanização agrícola</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="intro-text">Envie uma foto do óleo lubrificante do trator ou '
    "implemento e o modelo identifica o estado de conservação com base na cor, "
    "seguindo a escala ASTM.</p>",
    unsafe_allow_html=True,
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
    cor = cor_da_classe(top1_classe)

    with col2:
        st.markdown(
            f'<div class="result-card" style="background:{cor};">'
            f'<p class="classe">{top1_classe}</p>'
            f'<p class="conf">Confiança: {top1_conf:.1f}%</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        indices_ordenados = probs.data.argsort(descending=True)
        for idx in indices_ordenados:
            idx = int(idx)
            classe = nomes[idx]
            valor = float(probs.data[idx]) * 100
            st.markdown(
                f'<div class="prob-row"><span>{classe}</span>'
                f"<span>{valor:.1f}%</span></div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(valor / 100, 0.0), 1.0))
else:
    st.info("Aguardando o envio de uma imagem para classificar.")
