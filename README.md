# Classificador de Óleo Lubrificante

Site (Streamlit) que classifica o estado de um óleo lubrificante — **novo**,
**intermediário** ou **velho** — a partir de uma foto, usando um modelo YOLO
(Ultralytics, tarefa de classificação) treinado com fotos reais, seguindo a
escala de cor ASTM. Projeto derivado de um TCC sobre óleo lubrificante na
mecanização agrícola.

## Arquivos

- `app.py` — aplicação Streamlit (upload de imagem + classificação)
- `best.pt` — pesos do modelo YOLO treinado
- `requirements.txt` — dependências Python
- `.streamlit/config.toml` — tema visual do app

## Rodando localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501`.

## Publicando no Streamlit Community Cloud

1. Suba este repositório no GitHub (inclua o `best.pt`).
2. Acesse [share.streamlit.io](https://share.streamlit.io), conecte sua conta
   do GitHub e escolha este repositório.
3. Defina `app.py` como arquivo principal e clique em **Deploy**.

⚠️ **Atenção ao tamanho do `best.pt` (~57 MB):** o GitHub recomenda
[Git LFS](https://git-lfs.com/) para arquivos acima de 50 MB. Se o upload
comum falhar ou ficar lento, use:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes best.pt
git commit -m "Adiciona modelo via Git LFS"
```

## Como funciona

O `app.py` carrega o modelo com `ultralytics.YOLO("best.pt")` e roda a
inferência diretamente no servidor a cada imagem enviada — nenhuma conversão
de formato é necessária, é o modelo treinado original.
