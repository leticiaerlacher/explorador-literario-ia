import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from unidecode import unidecode
from gtts import gTTS 
import base64
import re

# 1. CONFIGURAÇÃO DA PÁGINA (Meta 6: Interface Lúdica)
st.set_page_config(page_title="IA Literária Científica", page_icon="🧪", layout="wide")

# Ancora para o scroll subir
st.markdown("<div id='inicio'></div>", unsafe_allow_html=True)

def normalizar_texto(txt):
    return unidecode(str(txt).lower().strip())

def get_base64_bin(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def gerar_audio(texto, titulo_livro):
    nome_limpo = re.sub(r'\W+', '', titulo_livro) 
    nome_arquivo = f"audio_{nome_limpo}.mp3"
    try:
        tts = gTTS(text=texto, lang='pt')
        tts.save(nome_arquivo)
        with open(nome_arquivo, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""<audio controls style="width: 100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>"""
            st.markdown(md, unsafe_allow_html=True)
        os.remove(nome_arquivo)
    except:
        st.error(f"Áudio de '{titulo_livro}' indisponível.")

# --- 2. INJEÇÃO DA IMAGEM NO FUNDO DA SIDEBAR ---
if os.path.exists("fundo_painel.png"):
    img_b64 = get_base64_bin("fundo_painel.png")
    st.sidebar.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{ background: transparent !important; }}
        [data-testid="stSidebar"] {{
            background-image: url("data:image/png;base64,{img_b64}");
            background-size: cover; background-position: center;
        }}
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] h2 {{
            color: #00D1FF !important; text-shadow: 2px 2px 4px black;
        }}
        </style>
        """, unsafe_allow_html=True
    )

# 3. ESTILO VISUAL DOS CARDS
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle, #0d1117 0%, #010409 100%); }
    .banner-container {
        width: 100%; border-radius: 25px; overflow: hidden; margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 209, 255, 0.2);
    }
    .stContainer {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        padding: 25px; margin-bottom: 25px;
    }
    .btn-navegacao {
        display: inline-block; padding: 10px 20px;
        background: rgba(0, 209, 255, 0.1); color: #00D1FF !important;
        border: 1px solid #00D1FF; border-radius: 12px;
        text-decoration: none; text-align: center; width: 100%; font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 4. CARREGAMENTO DOS DADOS (Meta 3)
try:
    df = pd.read_csv('Projeto_IA.csv', encoding='latin1', sep=';')
except:
    df = pd.read_csv('Projeto_IA.csv', encoding='utf-8', sep=',')
df = df.fillna('')

df['super_busca'] = (df['Titulo'] + " " + df['Autor'] + " " + df['Resumo']).apply(normalizar_texto)
tfidf = TfidfVectorizer(stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da'])
matrix = tfidf.fit_transform(df['super_busca'])

# 5. CONTEÚDO DA SIDEBAR (Filtros)
with st.sidebar:
    st.markdown("## 🛰️ Navegação Estelar")
    st.markdown("---")
    f_idade = st.selectbox("👤 Faixa Etária", ["Todos"] + sorted(df['Faixa_Etaria'].unique().tolist()))
    f_cat = st.selectbox("📖 Estilo de Leitura", ["Todas"] + sorted(df['Categoria_Literaria'].unique().tolist()))

if 'pagina_atual' not in st.session_state:
    st.session_state.pagina_atual = 0

# 6. BANNER PRINCIPAL
if os.path.exists("fundo_galaxia.png"):
    st.markdown('<div class="banner-container">', unsafe_allow_html=True)
    st.image("fundo_galaxia.png", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

pesquisa = st.text_input("", placeholder="🔍 O que vamos descobrir hoje no universo da leitura?")

# 7. LÓGICA DE RECOMENDAÇÃO (Meta 5 e 7)
df_final = df.copy()
if f_idade != "Todos": df_final = df_final[df_final['Faixa_Etaria'] == f_idade]
if f_cat != "Todas": df_final = df_final[df_final['Categoria_Literaria'] == f_cat]

if pesquisa:
    p_limpa = normalizar_texto(pesquisa)
    def calcular_peso(row):
        p = 0
        if p_limpa in normalizar_texto(row['Titulo']): p += 4.0
        if p_limpa in normalizar_texto(row['Tema_Cientifico']): p += 3.0
        return p
    df_final['peso'] = df_final.apply(calcular_peso, axis=1)
    sim = cosine_similarity(tfidf.transform([p_limpa]), matrix[df_final.index]).flatten()
    df_final['score'] = sim + df_final['peso']
    df_final = df_final[df_final['score'] > 0].sort_values('score', ascending=False)
    
    if not df_final.empty:
        with st.sidebar:
            st.divider()
            st.markdown("### ✨ Dica do Explorador:")
            
            # --- LÓGICA DE RECOMENDAÇÃO REFORÇADA ---
            tema_atual = df_final.iloc[0]['Tema_Cientifico']
            # Tenta achar outro livro do mesmo tema
            sug = df[(df['Tema_Cientifico'] == tema_atual) & (df['Titulo'] != df_final.iloc[0]['Titulo'])].head(1)
            
            # Se não achar por tema, pega o segundo melhor resultado da IA
            if sug.empty and len(df_final) > 1:
                sug = df_final.iloc[1:2]
            
            if not sug.empty:
                for _, s in sug.iterrows():
                    st.image(f"capas/{s['Arquivo_imagem']}", width=130)
                    st.info(f"Veja também: **{s['Titulo']}**")
            
            st.divider()
            st.markdown("### 📚 Você também pode gostar:")
            # Pega os próximos livros recomendados pela IA (evitando o que já apareceu na dica)
            sugestoes_extras = df_final.iloc[1:4] if sug.empty else df_final[df_final['Titulo'] != sug.iloc[0]['Titulo']].iloc[1:3]
            
            if not sugestoes_extras.empty:
                for _, s in sugestoes_extras.iterrows():
                    st.image(f"capas/{s['Arquivo_imagem']}", width=100)
                    st.write(f"**{s['Titulo']}**")

# 8. EXIBIÇÃO E PAGINAÇÃO
total_livros = len(df_final)
itens_por_pagina = 5
if total_livros > 0:
    inicio_idx = st.session_state.pagina_atual * itens_por_pagina
    fim_idx = inicio_idx + itens_por_pagina
    livros_p = df_final.iloc[inicio_idx:fim_idx]

    for _, livro in livros_p.iterrows():
        with st.container():
            c1, c2 = st.columns([1, 3])
            with c1:
                if os.path.exists(f"capas/{livro['Arquivo_imagem']}"):
                    st.image(f"capas/{livro['Arquivo_imagem']}", use_container_width=True)
            with c2:
                st.subheader(f"📖 {livro['Titulo']}")
                st.info(f"🧪 **Área Científica:** {livro['Tema_Cientifico']}")
                st.write(f"**Resumo:** {livro['Resumo']}")
                gerar_audio(livro['Resumo'], livro['Titulo'])
            st.divider()

    st.markdown("---")
    col_prev, col_cont, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.session_state.pagina_atual > 0:
            if st.button("Anterior"):
                st.session_state.pagina_atual -= 1
                st.rerun()
    with col_cont:
        total_p = (total_livros - 1) // itens_por_pagina + 1
        st.markdown(f"<p style='text-align:center; color:white;'>Página {st.session_state.pagina_atual + 1} de {total_p}</p>", unsafe_allow_html=True)
    with col_next:
        if fim_idx < total_livros:
            if st.button("Próximo"):
                st.session_state.pagina_atual += 1
                st.rerun()
else:
    st.info("Explore os mistérios do universo!")

st.markdown("<p style='text-align: center; color: #555; font-size: 10px;'>🚀 Letícia Erlacher | Iniciação Tecnológica | Multivix</p>", unsafe_allow_html=True)
