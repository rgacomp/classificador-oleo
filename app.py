import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Classificador de Óleo Lubrificante",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "best.pt"

# =========================
# VISUAL DA NOVA INTERFACE
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f4f7f5; }
.block-container { max-width: 1250px; padding-top: 0.8rem; padding-bottom: 2rem; }

.hero {
    background: linear-gradient(110deg, #063c2b 0%, #0b573c 58%, #164f3b 100%);
    border-radius: 0 0 22px 22px;
    padding: 32px 42px;
    color: white;
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}
.hero:after {
    content: ""; position:absolute; width:330px; height:330px; right:-100px; top:-170px;
    border-radius:50%; background:rgba(139,212,93,.16);
}
.hero-title { font-size:clamp(2rem,4vw,3.35rem); line-height:1.03; font-weight:800; margin:0; }
.hero-title span { color:#8bd45d; }
.hero-subtitle { font-size:1.05rem; margin-top:12px; color:#dceee5; }
.authors { margin-top:18px; font-size:.92rem; color:#d6e7df; }

.card {
    background:#fff; border:1px solid #e2e9e5; border-radius:18px; padding:24px;
    box-shadow:0 5px 18px rgba(8,50,34,.06); height:100%;
}
.card h2, .card h3 { color:#123d2f; margin-top:0; }
.muted { color:#63736b; }

.upload-box {
    border:2px dashed #bfd1c7; border-radius:16px; padding:28px 18px;
    text-align:center; background:#fbfdfc; margin:15px 0 16px;
}
.upload-icon { font-size:2.5rem; }
.upload-title { font-weight:700; color:#173d31; font-size:1.08rem; margin-top:8px; }

.result-good, .result-mid, .result-old {
    border-radius:14px; padding:18px; margin-bottom:12px;
}
.result-good { background:#edf8ef; border:1px solid #cde8d2; }
.result-mid { background:#fff8e8; border:1px solid #eedca7; }
.result-old { background:#f9eeee; border:1px solid #e7cccc; }
.result-title { font-size:1.35rem; font-weight:800; }
.metric { display:flex; justify-content:space-between; border-bottom:1px solid #edf0ee; padding:11px 0; }
.metric:last-child { border-bottom:0; }
.metric-label { color:#65736d; }
.metric-value { color:#173c30; font-weight:700; }

.astm-scale { display:grid; grid-template-columns:repeat(8,1fr); height:34px; border-radius:8px; overflow:hidden; margin-top:10px; }
.astm-scale div:nth-child(1){background:#f6dc75}.astm-scale div:nth-child(2){background:#eec34b}
.astm-scale div:nth-child(3){background:#dca33f}.astm-scale div:nth-child(4){background:#c47e42}
.astm-scale div:nth-child(5){background:#a85d43}.astm-scale div:nth-child(6){background:#7a463b}
.astm-scale div:nth-child(7){background:#51332e}.astm-scale div:nth-child(8){background:#241b1a}
.astm-numbers { display:grid; grid-template-columns:repeat(8,1fr); text-align:center; color:#68746e; font-size:.82rem; margin-top:5px; }

div.stButton > button {
    width:100%; border:0; border-radius:12px; min-height:50px;
    background:#11633f; color:white; font-weight:700; font-size:1rem;
}
div.stButton > button:hover { background:#0b4e32; color:white; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff; border:1px solid #e2e9e5 !important; border-radius:18px;
    box-shadow:0 5px 18px rgba(8,50,34,.06); padding:12px;
}

.app-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:15px; }
.app-item { text-align:center; color:#496158; font-size:.78rem; }
.app-circle { width:52px; height:52px; margin:auto auto 8px; display:grid; place-items:center; background:#eaf2ed; border-radius:50%; font-size:1.5rem; }

.footer { background:#063c2b; color:#d8e8e0; border-radius:18px; padding:22px; margin-top:28px; text-align:center; font-size:.85rem; }

/* Remove some Streamlit chrome around the uploader */
[data-testid="stFileUploader"] { margin-top:-8px; }
[data-testid="stFileUploaderDropzone"] { background:#fbfdfc; border-radius:12px; }

@media (max-width: 800px) {
    .hero { padding:25px 22px; }
    .card { padding:18px; }
    .app-grid { grid-template-columns:repeat(2,1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def carregar_modelo():
    return YOLO(MODEL_PATH)


def normalizar_classe(nome: str) -> str:
    return str(nome).strip().lower()


def estilo_resultado(nome: str):
    n = normalizar_classe(nome)
    if "novo" in n:
        return "result-good", "#176334", "Óleo em bom estado", "🟢"
    if "interm" in n or "medio" in n or "médio" in n:
        return "result-mid", "#8a6100", "Estado intermediário", "🟡"
    if "velho" in n or "usad" in n:
        return "result-old", "#8b3f2f", "Óleo com maior nível de uso", "🔴"
    return "result-mid", "#365b4b", str(nome).title(), "🔬"


# =========================
# CABEÇALHO
# =========================
st.markdown(
    """
<div class="hero">
    <div style="font-size:3rem; line-height:1;">🚜</div>
    <div class="hero-title">Classificador de<br><span>Óleo Lubrificante</span></div>
    <div class="hero-subtitle">Diagnóstico visual para mecanização agrícola</div>
    <div class="authors">👥 Rafael Gulli &nbsp;•&nbsp; Hiago Zanetoni</div>
</div>
""",
    unsafe_allow_html=True,
)

# Carrega o mesmo modelo real do projeto original
modelo = carregar_modelo()

left, right = st.columns([1, 1], gap="large")

# =========================
# COLUNA DE UPLOAD
# =========================
with left:
    st.markdown(
        """
<div class="card">
    <h2>📷 Envie a imagem do óleo</h2>
    <p class="muted">
        Selecione uma imagem do óleo lubrificante do trator ou implemento.
        O modelo treinado analisa a imagem e estima o estado de conservação.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    arquivo = st.file_uploader(
        "Escolher imagem",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    imagem = None
    if arquivo is not None:
        imagem = Image.open(arquivo).convert("RGB")
        st.image(imagem, caption="Imagem enviada", use_container_width=True)
        executar = st.button("🔬  CLASSIFICAR IMAGEM")
    else:
        executar = False

    st.markdown(
        """
<div class="card" style="margin-top:18px;">
    <h3>🚜 Aplicação</h3>
    <div class="app-grid">
      <div class="app-item"><div class="app-circle">🚜</div>Tratores</div>
      <div class="app-item"><div class="app-circle">🌾</div>Máquinas agrícolas</div>
      <div class="app-item"><div class="app-circle">⚙️</div>Implementos</div>
      <div class="app-item"><div class="app-circle">💧</div>Sistemas de lubrificação</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================
# COLUNA DE RESULTADO
# =========================
with right:
    box = st.container(border=True)
    box.markdown("<h2>📊 Resultado da análise</h2>", unsafe_allow_html=True)

    if arquivo is None:
        box.markdown(
            """
<div style="padding:48px 15px;text-align:center;color:#74827b;">
    <div style="font-size:3rem;">🔬</div>
    <h3>Aguardando uma imagem</h3>
    <p>Envie uma foto do óleo para visualizar o resultado da classificação.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    elif executar:
        with st.spinner("Analisando a imagem..."):
            resultado = modelo.predict(imagem, verbose=False)[0]

        nomes = resultado.names
        probs = resultado.probs
        top1_idx = int(probs.top1)
        top1_classe = str(nomes[top1_idx])
        top1_conf = float(probs.top1conf) * 100

        box_class, title_color, titulo, emoji = estilo_resultado(top1_classe)

        box.markdown(
            f"""
<div class="{box_class}">
    <div class="result-title" style="color:{title_color};">{emoji} {titulo}</div>
    <div style="margin-top:6px;color:#52645b;">Classe identificada pelo modelo: <b>{top1_classe}</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

        box.markdown(
            f"""
<div class="metric"><span class="metric-label">Classe do modelo</span><span class="metric-value">{top1_classe}</span></div>
<div class="metric"><span class="metric-label">Confiança</span><span class="metric-value">{top1_conf:.1f}%</span></div>
""",
            unsafe_allow_html=True,
        )

        box.markdown("<h3 style='margin-top:22px;'>Distribuição das classes</h3>", unsafe_allow_html=True)
        indices_ordenados = probs.data.argsort(descending=True)
        for idx in indices_ordenados:
            idx = int(idx)
            classe = str(nomes[idx])
            valor = float(probs.data[idx]) * 100
            box.markdown(
                f'<div class="metric"><span class="metric-label">{classe}</span><span class="metric-value">{valor:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
            box.progress(min(max(valor / 100, 0.0), 1.0))

        box.markdown(
            """
<div style="margin-top:22px;">
    <strong style="color:#173d31;">Escala visual de cores ASTM</strong>
    <p class="muted" style="font-size:.82rem;margin:4px 0 0;">
        Referência visual apresentada para contextualização; a classe final é determinada pelo modelo treinado.
    </p>
    <div class="astm-scale"><div></div><div></div><div></div><div></div><div></div><div></div><div></div><div></div></div>
    <div class="astm-numbers"><div>1</div><div>2</div><div>3</div><div>4</div><div>5</div><div>6</div><div>7</div><div>8</div></div>
    <div style="display:flex;justify-content:space-between;color:#78847f;font-size:.8rem;margin-top:5px;">
        <span>Mais claro</span><span>Mais escuro</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        box.markdown(
            """
<div style="padding:48px 15px;text-align:center;color:#74827b;">
    <div style="font-size:3rem;">📷</div>
    <h3>Imagem pronta para análise</h3>
    <p>Clique em <b>Classificar imagem</b> para executar o modelo.</p>
</div>
""",
            unsafe_allow_html=True,
        )


# =========================
# INFORMAÇÕES
# =========================
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(
        """
<div class="card">
  <h3>🌱 Sobre o projeto</h3>
  <p class="muted">
    Ferramenta de diagnóstico visual desenvolvida para auxiliar na análise
    do estado de conservação de óleo lubrificante utilizado em máquinas agrícolas.
  </p>
  <hr>
  <b>Desenvolvido por:</b><br>
  Rafael Gulli & Hiago Zanetoni
</div>
""",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
<div class="card" style="background:linear-gradient(145deg,#0a3c2b,#155a3e);color:white;">
  <h3 style="color:white;">⚙️ Manutenção é produtividade</h3>
  <p>
    O diagnóstico visual pode servir como ferramenta de apoio à tomada de decisão
    e ao acompanhamento das condições de lubrificação de máquinas agrícolas.
  </p>
  <br>🚜 Tecnologia aplicada ao agronegócio.
</div>
""",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
<div class="card">
  <h3>📚 Referência ASTM</h3>
  <p class="muted">
    A interface apresenta uma escala visual de referência para contextualizar
    a variação de tonalidade observada em óleos e derivados de petróleo.
  </p>
  <div style="margin-top:20px;padding:12px;border-radius:10px;background:#f2f6f3;">
    <b>Observação:</b> a classificação exibida é a saída do modelo treinado,
    e não substitui análises laboratoriais.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="footer">
    🚜 &nbsp; Classificador de Óleo Lubrificante
    &nbsp;&nbsp; | &nbsp;&nbsp; Rafael Gulli & Hiago Zanetoni
</div>
""",
    unsafe_allow_html=True,
)
