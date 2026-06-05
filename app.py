import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# configuração da página
st.set_page_config(
    page_title="Análise de Sentimentos - Livros da Amazon 2023",
    page_icon="📚",
    layout="centered"
)

# desativar spellcheck (tracejado vermelho) usando MutationObserver para reagir a renderizações do streamlit
components.html(
    """
<script>
const doc = window.parent.document;
const disableSpellcheck = () => {
    doc.querySelectorAll('textarea').forEach(el => el.setAttribute('spellcheck', 'false'));
};
// Executa logo de cara
disableSpellcheck();
// Cria um observer para desativar sempre que o Streamlit re-renderizar a caixa de texto
const observer = new MutationObserver(disableSpellcheck);
observer.observe(doc.body, { childList: true, subtree: true });
</script>
""",
    height=0,
    width=0,
)

# estilizando a página com CSS
st.markdown("""
<style>
    .main-title {
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .positivo {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
    }
    .negativo {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
    }
    .neutro {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeeba !important;
    }
</style>
""", unsafe_allow_html=True)

# criando um Cache -> carregamento do modelo ser mais rápido
@st.cache_resource
def load_model():
    model_path = "./modelo_sentimentos_livros"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Verifique se a pasta '{model_path}' existe e contém os arquivos do modelo.")
        st.stop()

tokenizer, model = load_model()
label_map = {0: 'Negativo', 1: 'Neutro', 2: 'Positivo'}

def classificar_comentario(frase):
    inputs = tokenizer(
        frase,
        return_tensors="pt",
        truncation=True, 
        padding=True, 
        max_length=128
    )

    # device agnostic code    
    inputs = inputs.to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    probabilidades = torch.softmax(logits, dim=1)[0]
    predicao = torch.argmax(logits, dim=1).item()
    
    return predicao, probabilidades

# interface do App
st.markdown("<h1 class='main-title'>📚 Analisador de Sentimentos de Livros</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    # mostrando a capa de um livro como exemplo
    st.image("capa_livro.jpg", caption="Capa Ilustrativa", use_container_width=True)

with col2:
    st.markdown("### Deixe seu comentário sobre o livro")
    user_input = st.text_area("O que você achou da leitura?", height=150, placeholder="Comente aqui sobre o livro...")
    
    if st.button("Analisar Sentimento", type="primary", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Por favor, digite um comentário antes de analisar.")
        else:
            with st.spinner("Analisando o sentimento..."):
                predicao, probs = classificar_comentario(user_input)
                sentimento = label_map[predicao]
                confianca = probs[predicao].item() * 100
                
                # a classe css é baseada no sentimento
                css_class = sentimento.lower()
                
                # mostrando o resultado
                st.markdown(f"""
                <div class="result-box {css_class}">
                    <h2>Sentimento: {sentimento}</h2>
                    <p>Confiança: {confianca:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # explainer pra mostrar a probabilidade de cada uma das classes
                with st.expander("Ver detalhes das probabilidades"):
                    st.write(f"👎 Negativo: {probs[0].item()*100:.1f}%")
                    st.write(f"😐 Neutro: {probs[1].item()*100:.1f}%")
                    st.write(f"👍 Positivo: {probs[2].item()*100:.1f}%")
