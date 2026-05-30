import streamlit as st
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

# Suppress HuggingFace unauthenticated request warnings
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_pages" not in st.session_state:
    st.session_state.doc_pages = 0
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = 0

D = st.session_state.dark_mode

# ── color tokens ──────────────────────────────────────────────────────────────
if D:
    BG       = "#0E1117"
    SURF     = "#161B22"
    SURF2    = "#1C2333"
    BORDER   = "#30363D"
    TEXT     = "#E6EDF3"
    TEXT2    = "#8B949E"
    ACCENT   = "#58A6FF"
    ACCENT_D = "#1F6FEB"
    USER_BG  = "#1F6FEB"
    USER_TXT = "#FFFFFF"
    BOT_BG   = "#1C2333"
    BOT_TXT  = "#E6EDF3"
    TAG_BG   = "#1F2D45"
    TAG_TXT  = "#58A6FF"
    TAG_BDR  = "#1F6FEB"
    UPL_BG   = "#161B22"
    UPL_BDR  = "#30363D"
    UPL_TXT  = "#8B949E"
    ICON_SUN = "☀️"
    TOGGLE_LABEL = "☀️"
else:
    BG       = "#F6F8FA"
    SURF     = "#FFFFFF"
    SURF2    = "#F0F4F8"
    BORDER   = "#D0D7DE"
    TEXT     = "#1C2333"
    TEXT2    = "#6E7781"
    ACCENT   = "#0969DA"
    ACCENT_D = "#0550AE"
    USER_BG  = "#0969DA"
    USER_TXT = "#FFFFFF"
    BOT_BG   = "#FFFFFF"
    BOT_TXT  = "#1C2333"
    TAG_BG   = "#DDF4FF"
    TAG_TXT  = "#0550AE"
    TAG_BDR  = "#B6D4FB"
    UPL_BG   = "#FFFFFF"
    UPL_BDR  = "#D0D7DE"
    UPL_TXT  = "#6E7781"
    TOGGLE_LABEL = "🌙"

LOGO_SVG = f"""<svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="2" y="1" width="13" height="17" rx="2" stroke="white" stroke-width="1.5"/>
  <line x1="5" y1="5.5" x2="12" y2="5.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="5" y1="8.5" x2="12" y2="8.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
  <line x1="5" y1="11.5" x2="9" y2="11.5" stroke="white" stroke-width="1.5" stroke-linecap="round"/>
  <circle cx="16" cy="15" r="4.5" fill="{ACCENT_D}" stroke="white" stroke-width="1.5"/>
  <path d="M14.2 15l1.2 1.2 2.4-2.4" stroke="white" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    background-color: {BG} !important;
    font-family: 'Inter', sans-serif !important;
    color: {TEXT} !important;
}}

/* hide default streamlit header */
header[data-testid="stHeader"] {{ background: {BG} !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: {SURF} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] .stMarkdown p {{ color: {TEXT2} !important; }}

/* file uploader — fully themed */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: {UPL_BG} !important;
    border: 1.5px dashed {UPL_BDR} !important;
    border-radius: 10px !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
    color: {UPL_TXT} !important;
    background: transparent !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
    border: 1px solid {UPL_BDR} !important;
    background: {SURF2} !important;
    color: {TEXT} !important;
    border-radius: 6px !important;
}}

/* theme toggle button — small, top right of sidebar */
.theme-btn button {{
    background: {SURF2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
    font-size: 18px !important;
    padding: 4px 10px !important;
    width: auto !important;
    float: right;
}}

/* clear btn */
.clear-btn button {{
    background: transparent !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT2} !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    padding: 5px 12px !important;
    width: 100% !important;
}}

/* chat input */
.stChatInputContainer {{
    background: {SURF} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
}}
.stChatInputContainer textarea {{
    color: {TEXT} !important;
    background: {SURF} !important;
}}

/* slider */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background: {ACCENT} !important;
    border-color: {ACCENT} !important;
}}

/* toggle */
[data-testid="stToggle"] label {{ color: {TEXT} !important; }}

/* expander */
.stExpander {{
    background: {SURF2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}

/* metrics */
div[data-testid="stMetricValue"] {{
    color: {ACCENT} !important;
    font-size: 20px !important;
    font-weight: 600 !important;
}}
div[data-testid="stMetricLabel"] {{ color: {TEXT2} !important; font-size: 11px !important; }}
div[data-testid="metric-container"] {{
    background: {SURF} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}}

.stDivider {{ border-color: {BORDER} !important; }}
.stMarkdown p {{ color: {TEXT} !important; }}

/* logo */
.logo-wrap {{ display:flex; align-items:center; gap:12px; padding:4px 0 12px; }}
.logo-icon {{
    width:40px; height:40px; background:{ACCENT_D};
    border-radius:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}}
.logo-name {{ font-size:17px; font-weight:600; color:{TEXT}; letter-spacing:-0.3px; }}
.logo-tag  {{ font-size:11px; color:{TEXT2}; margin-top:1px; }}

/* doc card */
.doc-card {{
    background:{SURF2}; border:1px solid {BORDER};
    border-radius:10px; padding:12px 14px; margin-top:8px;
}}
.doc-name {{ font-size:13px; font-weight:500; color:{TEXT}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.doc-meta {{ font-size:11px; color:{TEXT2}; margin-top:3px; }}
.progress-track {{ height:3px; background:{BORDER}; border-radius:2px; margin-top:8px; overflow:hidden; }}
.progress-fill  {{ height:3px; background:{ACCENT}; border-radius:2px; width:100%; }}

/* chat bubbles */
.chat-user {{ display:flex; justify-content:flex-end; margin:8px 0; }}
.chat-user-bubble {{
    background:{USER_BG}; color:{USER_TXT};
    border-radius:16px 16px 4px 16px;
    padding:10px 14px; max-width:72%;
    font-size:14px; line-height:1.55;
}}
.chat-bot {{ display:flex; gap:10px; margin:8px 0; align-items:flex-start; }}
.bot-avatar {{
    width:30px; height:30px; background:{ACCENT_D};
    border-radius:8px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
}}
.chat-bot-bubble {{
    background:{BOT_BG}; color:{BOT_TXT};
    border:1px solid {BORDER};
    border-radius:4px 16px 16px 16px;
    padding:10px 14px; max-width:82%;
    font-size:14px; line-height:1.6;
}}
.src-tag {{
    display:inline-block; background:{TAG_BG}; color:{TAG_TXT};
    border:1px solid {TAG_BDR}; border-radius:20px;
    font-size:11px; padding:2px 10px; margin:2px 3px 2px 0; font-weight:500;
}}

/* empty state */
.empty-state {{
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; padding:5rem 2rem; text-align:center;
}}
.empty-icon {{
    width:64px; height:64px; background:{SURF2};
    border:1px solid {BORDER}; border-radius:16px;
    display:flex; align-items:center; justify-content:center; margin-bottom:1.5rem;
}}
.empty-title {{ font-size:18px; font-weight:600; color:{TEXT}; margin-bottom:8px; }}
.empty-sub   {{ font-size:14px; color:{TEXT2}; line-height:1.6; max-width:340px; }}

/* step rows */
.step-row {{ display:flex; gap:10px; align-items:flex-start; margin-bottom:8px; }}
.step-num {{
    width:20px; height:20px; border-radius:50%;
    background:{TAG_BG}; color:{TAG_TXT};
    font-size:10px; font-weight:600;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
    border:1px solid {TAG_BDR};
}}
.step-txt {{ font-size:12px; color:{TEXT2}; padding-top:2px; line-height:1.5; }}

/* section labels */
.sec-label {{
    font-size:10px; font-weight:600; color:{TEXT2};
    text-transform:uppercase; letter-spacing:.07em; margin-bottom:8px;
}}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # logo + theme toggle on same row
    col_logo, col_toggle = st.columns([4, 1])
    with col_logo:
        st.markdown(f"""
        <div class="logo-wrap">
          <div class="logo-icon">{LOGO_SVG}</div>
          <div>
            <div class="logo-name">DocuMind AI</div>
            <div class="logo-tag">Document Intelligence</div>
          </div>
        </div>""", unsafe_allow_html=True)
    with col_toggle:
        st.markdown('<div class="theme-btn" style="padding-top:14px">', unsafe_allow_html=True)
        if st.button(TOGGLE_LABEL, key="theme_toggle"):
            st.session_state.dark_mode = not D
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown(f'<div class="sec-label">Document</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if st.session_state.doc_name:
        st.markdown(f"""
        <div class="doc-card">
          <div class="doc-name">{st.session_state.doc_name}</div>
          <div class="doc-meta">{st.session_state.doc_pages} pages · {st.session_state.doc_chunks} chunks</div>
          <div class="progress-track"><div class="progress-fill"></div></div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown(f'<div class="sec-label">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Sources retrieved", 2, 6, 3)
    show_sources = st.toggle("Show source pages", value=True)
    show_scores  = st.toggle("Show relevance %", value=True)

    st.divider()

    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown(f'<div class="sec-label">How it works</div>', unsafe_allow_html=True)
    for i, s in enumerate([
        "Upload any PDF document",
        "AI chunks it into 1000-char pieces",
        "Each chunk becomes a vector",
        "Question finds closest chunks",
        "LLM answers from retrieved context"
    ]):
        st.markdown(f"""
        <div class="step-row">
          <div class="step-num">{i+1}</div>
          <div class="step-txt">{s}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:11px;color:{TEXT2};margin-top:12px;text-align:center">DocuMind AI · Document Intelligence</div>', unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    try:
        api_key = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
    return ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_tokens=1024
    )

@st.cache_resource(show_spinner="Building index...")
def build_vectorstore(_pages, doc_name):
    # Filter out empty pages
    valid_pages = [p for p in _pages if p.page_content.strip()]
    if not valid_pages:
        st.error("No readable text found in this PDF.")
        return None, 0
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(valid_pages)
    vs = FAISS.from_documents(chunks, load_embeddings())
    return vs, len(chunks)

def fmt_docs(docs):
    return "\n\n".join([f"[Page {d.metadata.get('page','?')}]: {d.page_content}" for d in docs])

# ── UPLOAD HANDLER ────────────────────────────────────────────────────────────
if uploaded and uploaded.name != st.session_state.doc_name:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    with st.spinner(f"Indexing {uploaded.name}..."):
        pages = PyPDFLoader(tmp_path).load()
        vs, nc = build_vectorstore(pages, uploaded.name)
        st.session_state.vectorstore = vs
        st.session_state.doc_name    = uploaded.name
        st.session_state.doc_pages   = len(pages)
        st.session_state.doc_chunks  = nc
        st.session_state.messages    = []
    os.unlink(tmp_path)
    st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-icon">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="{ACCENT}" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="12" y1="11" x2="12" y2="17"/>
          <line x1="9" y1="14" x2="15" y2="14"/>
        </svg>
      </div>
      <div class="empty-title">Upload a PDF to get started</div>
      <div class="empty-sub">Ask questions about research papers, textbooks, contracts, or lecture notes. Every answer includes exact page citations.</div>
    </div>""", unsafe_allow_html=True)

else:
    # stats row
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages", st.session_state.doc_pages)
    c2.metric("Chunks indexed", st.session_state.doc_chunks)
    c3.metric("Questions asked", len([m for m in st.session_state.messages if m["role"]=="user"]))

    st.divider()

    # chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        else:
            tags = ""
            if show_sources and "sources" in msg:
                for src in msg["sources"]:
                    rel = f" · {src['rel']}%" if show_scores else ""
                    tags += f'<span class="src-tag">Page {src["page"]}{rel}</span>'
            src_html = f'<div style="margin-top:8px">{tags}</div>' if tags else ""
            st.markdown(f"""
            <div class="chat-bot">
              <div class="bot-avatar">{LOGO_SVG}</div>
              <div class="chat-bot-bubble">{msg["content"]}{src_html}</div>
            </div>""", unsafe_allow_html=True)

    # input
    if question := st.chat_input("Ask anything about your document..."):
        st.session_state.messages.append({"role":"user","content":question})
        st.markdown(f'<div class="chat-user"><div class="chat-user-bubble">{question}</div></div>', unsafe_allow_html=True)

        results = st.session_state.vectorstore.similarity_search_with_score(question, k=top_k)
        docs   = [r[0] for r in results]
        scores = [r[1] for r in results]
        context = fmt_docs(docs)

        prompt = PromptTemplate.from_template("""You are DocuMind AI, a precise document assistant.
Answer using only the context below. Cite page numbers inline like (Page X).
If not found, say "I couldn't find that in this document."

Context:
{context}

Question: {question}

Answer:""")

        chain = prompt | load_llm() | StrOutputParser()

        with st.spinner("Thinking..."):
            try:
                response = "".join(chain.stream({"context": context, "question": question}))
            except Exception as e:
                err = str(e).lower()
                if "rate" in err:
                    st.error("Rate limit hit — please wait 10 seconds and try again.")
                elif "auth" in err or "key" in err:
                    st.error("API key error — check Streamlit secrets.")
                else:
                    st.error(f"Error: {str(e)}")
                st.stop()

        sources = [{"page": docs[i].metadata.get("page","?"),
                    "rel": max(0, min(100, int((1-scores[i])*100)))}
                   for i in range(len(docs))]

        tags = ""
        if show_sources:
            for src in sources:
                rel = f" · {src['rel']}%" if show_scores else ""
                tags += f'<span class="src-tag">Page {src["page"]}{rel}</span>'
        src_html = f'<div style="margin-top:8px">{tags}</div>' if tags else ""

        st.markdown(f"""
        <div class="chat-bot">
          <div class="bot-avatar">{LOGO_SVG}</div>
          <div class="chat-bot-bubble">{response}{src_html}</div>
        </div>""", unsafe_allow_html=True)

        st.session_state.messages.append({"role":"assistant","content":response,"sources":sources})