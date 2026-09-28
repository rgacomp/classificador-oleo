# Classificador de Óleo Lubrificante

Interface renovada do projeto de Rafael Gulli e Hiago Zanetoni, mantendo o modelo YOLO real (`best.pt`) usado no site original.

## Arquivos principais

- `app.py` — nova interface + integração com o modelo real.
- `best.pt` — pesos do modelo YOLO original.
- `requirements.txt` — dependências.
- `packages.txt` — pacotes do sistema necessários.
- `.streamlit/config.toml` — tema e limite de upload.

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Use `app.py` como arquivo principal. Mantenha `best.pt` no repositório. Como o modelo tem aproximadamente 57 MB, Git LFS pode ser necessário dependendo da forma de publicação no GitHub.

## Observação

A escala ASTM mostrada na interface é uma referência visual. A classe e a confiança apresentadas no resultado vêm diretamente do modelo YOLO treinado e não foram inventadas pela interface.
