import streamlit as st
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from agent import get_app, DOCUMENTS, CapstoneState
from sentence_transformers import SentenceTransformer
import chromadb

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShopFAQ — E-Commerce Store", layout="wide", page_icon="🛍️")

# ── Inject all CSS + HTML ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0d0d0d !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }

[data-testid="stAppViewBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── NAVBAR ── */
.navbar {
    width: 100%; background: #0d0d0d;
    border-bottom: 1px solid #2a2a2a;
    padding: 18px 60px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
}
.navbar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 26px; font-weight: 900;
    color: #e8c97e; letter-spacing: -0.5px;
}
.navbar-links {
    display: flex; gap: 36px;
    font-size: 13px; font-weight: 500;
    color: #888; letter-spacing: 1px; text-transform: uppercase;
}
.navbar-links span { cursor: pointer; transition: color 0.2s; }
.navbar-links span:hover { color: #f0ede6; }
.navbar-right { display: flex; gap: 20px; align-items: center; }
.cart-icon {
    font-size: 20px; cursor: pointer;
    position: relative;
}
.cart-badge {
    position: absolute; top: -6px; right: -8px;
    background: #e8c97e; color: #0d0d0d;
    font-size: 9px; font-weight: 700;
    border-radius: 50%; width: 16px; height: 16px;
    display: flex; align-items: center; justify-content: center;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, #111 0%, #1a1a1a 50%, #0f0f0f 100%);
    padding: 100px 60px;
    display: flex; align-items: center; justify-content: space-between;
    min-height: 520px; position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(232,201,126,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-text { max-width: 560px; z-index: 1; }
.hero-eyebrow {
    font-size: 11px; font-weight: 600;
    color: #e8c97e; letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 20px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(42px, 5vw, 72px);
    font-weight: 900; line-height: 1.05;
    color: #f0ede6; margin-bottom: 24px;
}
.hero-title span { color: #e8c97e; }
.hero-sub {
    font-size: 16px; line-height: 1.7;
    color: #888; margin-bottom: 40px; max-width: 440px;
}
.hero-cta {
    display: inline-block;
    background: #e8c97e; color: #0d0d0d;
    font-size: 13px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 16px 36px; border-radius: 2px;
    cursor: pointer; transition: all 0.2s;
    border: 2px solid #e8c97e;
}
.hero-cta:hover { background: transparent; color: #e8c97e; }
.hero-cta-ghost {
    display: inline-block; margin-left: 16px;
    border: 2px solid #333; color: #888;
    font-size: 13px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 16px 36px; border-radius: 2px; cursor: pointer;
    transition: all 0.2s;
}
.hero-cta-ghost:hover { border-color: #888; color: #f0ede6; }
.hero-visual {
    position: relative; z-index: 1;
    display: flex; align-items: center; justify-content: center;
}
.hero-image-placeholder {
    width: 380px; height: 380px;
    border: 1px solid #2a2a2a; border-radius: 4px;
    background: linear-gradient(145deg, #181818, #141414);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px; color: #333; font-size: 80px;
}
.hero-image-placeholder p {
    font-size: 13px; color: #444;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 1px; text-transform: uppercase;
}

/* ── MARQUEE BANNER ── */
.marquee-wrap {
    background: #e8c97e; padding: 12px 0;
    overflow: hidden; white-space: nowrap;
}
.marquee-inner {
    display: inline-block;
    animation: marquee 28s linear infinite;
    font-size: 12px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #0d0d0d;
}
@keyframes marquee {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

/* ── CATEGORIES ── */
.section { padding: 80px 60px; }
.section-header {
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 40px; border-bottom: 1px solid #1e1e1e; padding-bottom: 20px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px; font-weight: 700; color: #f0ede6;
}
.section-link {
    font-size: 12px; font-weight: 600;
    color: #e8c97e; letter-spacing: 1.5px; text-transform: uppercase;
    cursor: pointer;
}
.categories-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
}
.category-card {
    background: #141414; border: 1px solid #1e1e1e;
    border-radius: 3px; padding: 36px 24px;
    text-align: center; cursor: pointer;
    transition: all 0.25s;
}
.category-card:hover { background: #1a1a1a; border-color: #e8c97e; transform: translateY(-2px); }
.category-icon { font-size: 36px; margin-bottom: 14px; }
.category-name {
    font-family: 'Playfair Display', serif;
    font-size: 17px; font-weight: 700;
    color: #f0ede6; margin-bottom: 6px;
}
.category-count { font-size: 12px; color: #555; letter-spacing: 0.5px; }

/* ── FEATURED PRODUCTS ── */
.products-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;
}
.product-card {
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 3px; overflow: hidden;
    cursor: pointer; transition: all 0.25s;
}
.product-card:hover { border-color: #333; transform: translateY(-3px); box-shadow: 0 20px 40px rgba(0,0,0,0.5); }
.product-image {
    height: 200px; background: #181818;
    display: flex; align-items: center; justify-content: center;
    font-size: 56px; border-bottom: 1px solid #1e1e1e;
    position: relative;
}
.product-badge {
    position: absolute; top: 14px; left: 14px;
    background: #e8c97e; color: #0d0d0d;
    font-size: 9px; font-weight: 800;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 4px 10px; border-radius: 1px;
}
.product-info { padding: 20px; }
.product-category {
    font-size: 10px; font-weight: 600;
    color: #555; letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 8px;
}
.product-name {
    font-family: 'Playfair Display', serif;
    font-size: 18px; font-weight: 700;
    color: #f0ede6; margin-bottom: 12px; line-height: 1.3;
}
.product-footer {
    display: flex; align-items: center; justify-content: space-between;
}
.product-price { font-size: 20px; font-weight: 700; color: #e8c97e; }
.product-old-price {
    font-size: 13px; color: #444;
    text-decoration: line-through; margin-left: 8px;
}
.add-to-cart {
    background: transparent; border: 1px solid #2a2a2a;
    color: #888; font-size: 11px; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase;
    padding: 8px 16px; border-radius: 2px; cursor: pointer;
    transition: all 0.2s;
}
.add-to-cart:hover { border-color: #e8c97e; color: #e8c97e; }

/* ── PROMO STRIP ── */
.promo-strip {
    background: #111; border-top: 1px solid #1e1e1e; border-bottom: 1px solid #1e1e1e;
    padding: 50px 60px;
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 0;
}
.promo-item {
    padding: 20px 40px; text-align: center;
    border-right: 1px solid #1e1e1e;
}
.promo-item:last-child { border-right: none; }
.promo-icon { font-size: 28px; margin-bottom: 12px; }
.promo-title {
    font-family: 'Playfair Display', serif;
    font-size: 17px; font-weight: 700;
    color: #f0ede6; margin-bottom: 6px;
}
.promo-desc { font-size: 13px; color: #555; line-height: 1.5; }

/* ── FOOTER ── */
.footer {
    background: #080808; border-top: 1px solid #1a1a1a;
    padding: 60px 60px 30px; 
}
.footer-grid {
    display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 40px;
    margin-bottom: 50px;
}
.footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 22px; font-weight: 900;
    color: #e8c97e; margin-bottom: 14px;
}
.footer-desc { font-size: 13px; color: #444; line-height: 1.7; max-width: 240px; }
.footer-col-title {
    font-size: 11px; font-weight: 700;
    color: #f0ede6; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 18px;
}
.footer-links { display: flex; flex-direction: column; gap: 10px; }
.footer-links a {
    font-size: 13px; color: #444; text-decoration: none;
    cursor: pointer; transition: color 0.2s;
}
.footer-links a:hover { color: #f0ede6; }
.footer-bottom {
    border-top: 1px solid #1a1a1a; padding-top: 24px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 12px; color: #333;
}
</style>

<!-- NAVBAR -->
<div class="navbar">
  <div class="navbar-logo">🛍️ ShopFAQ</div>
  <div class="navbar-links">
    <span>New Arrivals</span>
    <span>Electronics</span>
    <span>Apparel</span>
    <span>Home & Living</span>
    <span>Sale</span>
  </div>
  <div class="navbar-right">
    <span style="font-size:13px;color:#888;cursor:pointer;">🔍</span>
    <span style="font-size:13px;color:#888;cursor:pointer;">👤</span>
    <span class="cart-icon">🛒 <span class="cart-badge">3</span></span>
  </div>
</div>

<!-- HERO -->
<div class="hero">
  <div class="hero-text">
    <div class="hero-eyebrow">New Season Collection</div>
    <h1 class="hero-title">Premium Products,<br><span>Exceptional</span><br>Service.</h1>
    <p class="hero-sub">Discover curated electronics, fashion, and home essentials — shipped fast, backed by our no-fuss return policy.</p>
    <a class="hero-cta">Shop Now</a>
    <a class="hero-cta-ghost">Explore Deals</a>
  </div>
  <div class="hero-visual">
    <div class="hero-image-placeholder">
      🛍️
      <p>New Collection 2025</p>
    </div>
  </div>
</div>

<!-- MARQUEE -->
<div class="marquee-wrap">
  <div class="marquee-inner">
    &nbsp;&nbsp;&nbsp;FREE SHIPPING ON ORDERS OVER $50 &nbsp;•&nbsp; 30-DAY RETURNS &nbsp;•&nbsp; 1-YEAR WARRANTY ON ELECTRONICS &nbsp;•&nbsp; SHIP TO 50+ COUNTRIES &nbsp;•&nbsp; SECURE CHECKOUT &nbsp;•&nbsp; FREE SHIPPING ON ORDERS OVER $50 &nbsp;•&nbsp; 30-DAY RETURNS &nbsp;•&nbsp; 1-YEAR WARRANTY ON ELECTRONICS &nbsp;•&nbsp; SHIP TO 50+ COUNTRIES &nbsp;•&nbsp; SECURE CHECKOUT &nbsp;&nbsp;&nbsp;
  </div>
</div>

<!-- CATEGORIES -->
<div class="section">
  <div class="section-header">
    <div class="section-title">Browse Categories</div>
    <div class="section-link">View All →</div>
  </div>
  <div class="categories-grid">
    <div class="category-card"><div class="category-icon">💻</div><div class="category-name">Electronics</div><div class="category-count">240+ items</div></div>
    <div class="category-card"><div class="category-icon">👗</div><div class="category-name">Apparel</div><div class="category-count">580+ items</div></div>
    <div class="category-card"><div class="category-icon">🏠</div><div class="category-name">Home & Living</div><div class="category-count">320+ items</div></div>
    <div class="category-card"><div class="category-icon">🎧</div><div class="category-name">Accessories</div><div class="category-count">190+ items</div></div>
  </div>
</div>

<!-- FEATURED PRODUCTS -->
<div class="section" style="padding-top:0;">
  <div class="section-header">
    <div class="section-title">Featured Products</div>
    <div class="section-link">Shop All →</div>
  </div>
  <div class="products-grid">
    <div class="product-card">
      <div class="product-image">💻<div class="product-badge">Best Seller</div></div>
      <div class="product-info">
        <div class="product-category">Electronics</div>
        <div class="product-name">UltraBook Pro 15"</div>
        <div class="product-footer">
          <div><span class="product-price">$899</span><span class="product-old-price">$1,099</span></div>
          <button class="add-to-cart">+ Add</button>
        </div>
      </div>
    </div>
    <div class="product-card">
      <div class="product-image">🎧<div class="product-badge">New</div></div>
      <div class="product-info">
        <div class="product-category">Accessories</div>
        <div class="product-name">NoiseShield Pro Headphones</div>
        <div class="product-footer">
          <div><span class="product-price">$199</span><span class="product-old-price">$249</span></div>
          <button class="add-to-cart">+ Add</button>
        </div>
      </div>
    </div>
    <div class="product-card">
      <div class="product-image">📱</div>
      <div class="product-info">
        <div class="product-category">Electronics</div>
        <div class="product-name">SmartPhone X12 Ultra</div>
        <div class="product-footer">
          <div><span class="product-price">$649</span></div>
          <button class="add-to-cart">+ Add</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- PROMO STRIP -->
<div class="promo-strip">
  <div class="promo-item"><div class="promo-icon">🚚</div><div class="promo-title">Free Shipping</div><div class="promo-desc">On all orders above $50. Standard 3-5 day delivery.</div></div>
  <div class="promo-item"><div class="promo-icon">↩️</div><div class="promo-title">Easy Returns</div><div class="promo-desc">30-day no-fuss returns. We'll handle the label.</div></div>
  <div class="promo-item"><div class="promo-icon">🛡️</div><div class="promo-title">1-Year Warranty</div><div class="promo-desc">All electronics backed by manufacturer warranty.</div></div>
</div>

<!-- FOOTER -->
<div class="footer">
  <div class="footer-grid">
    <div>
      <div class="footer-brand">ShopFAQ</div>
      <div class="footer-desc">Your destination for premium electronics, fashion, and lifestyle products — delivered with care.</div>
    </div>
    <div>
      <div class="footer-col-title">Support</div>
      <div class="footer-links">
        <a>Track Order</a><a>Return Portal</a><a>Warranty Claims</a><a>Contact Us</a>
      </div>
    </div>
    <div>
      <div class="footer-col-title">Company</div>
      <div class="footer-links">
        <a>About</a><a>Careers</a><a>Blog</a><a>Press</a>
      </div>
    </div>
    <div>
      <div class="footer-col-title">Legal</div>
      <div class="footer-links">
        <a>Privacy Policy</a><a>Terms of Service</a><a>Cookie Policy</a>
      </div>
    </div>
  </div>
  <div class="footer-bottom">
    <span>© 2025 ShopFAQ. All rights reserved.</span>
    <span>support@shopfaq.example.com &nbsp;|&nbsp; 1-800-SHOP-FAQ</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Backend: load agent ───────────────────────────────────────────────────────
@st.cache_resource
def load_agent_and_kb():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")
    texts = [d["text"] for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )
    app = get_app(collection=collection, embedder=embedder)
    return app, collection

app, collection = load_agent_and_kb()

# Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# ── CHATBOT UI: NATIVE STREAMLIT WITH PREMIUM STYLING ────────────────────────
st.markdown("""
<style>
/* Bubble button positioning */
div.stButton > button[kind="secondary"] {
    position: fixed !important;
    bottom: 32px !important;
    right: 32px !important;
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    background: #e8c97e !important;
    color: #0d0d0d !important;
    font-size: 26px !important;
    z-index: 9998 !important;
    box-shadow: 0 8px 32px rgba(232,201,126,0.3) !important;
    border: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.25s !important;
}
div.stButton > button[kind="secondary"]:hover {
    transform: scale(1.1) !important;
    box-shadow: 0 12px 40px rgba(232,201,126,0.45) !important;
}

/* Chat window positioning */
[data-testid="stVerticalBlock"] > div:has(.chat-container-inner) {
    position: fixed !important;
    bottom: 106px !important;
    right: 32px !important;
    width: 380px !important;
    max-height: 540px !important;
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    z-index: 9999 !important;
    display: flex !important;
    flex-direction: column !important;
    box-shadow: 0 24px 80px rgba(0,0,0,0.8) !important;
    overflow: hidden !important;
}

.chat-container-inner { padding: 0 !important; }

/* Custom Chat Style inside the container */
.stChatMessage { border: none !important; background: transparent !important; padding: 10px 14px !important; }
.stChatMessage [data-testid="stChatMessageAvatar"] { width: 32px !important; height: 32px !important; }
.stChatMessage [data-testid="stChatMessageContent"] { 
    font-size: 13.5px !important; 
    line-height: 1.55 !important;
    color: #d0cdc6 !important;
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    border-bottom-left-radius: 3px !important;
}
.stChatMessage[data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] {
    background: #e8c97e !important;
    color: #0d0d0d !important;
    border: none !important;
    border-bottom-left-radius: 12px !important;
    border-bottom-right-radius: 3px !important;
}
/* Ensure chat input stays clean */
[data-testid="stChatInput"] { position: relative !important; z-index: 10000 !important; }
</style>
""", unsafe_allow_html=True)

# THE BUBBLE BUTTON
bubble_icon = "✕" if st.session_state.chat_open else "💬"
if st.button(bubble_icon, key="bubble_trigger"):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

# THE CHAT WINDOW
if st.session_state.chat_open:
    with st.container():
        st.markdown('<div class="chat-container-inner"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0d0d0d; border-bottom:1px solid #1e1e1e; padding:16px 20px; display:flex; align-items:center; gap:12px;">
          <div style="width:38px; height:38px; background:#e8c97e; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:18px;">🤖</div>
          <div>
            <div style="font-family:'Playfair Display',serif; font-size:15px; font-weight:700; color:#f0ede6;">ShopFAQ Assistant</div>
            <div style="font-size:11px; color:#4caf7d; display:flex; align-items:center; gap:5px;"><div style="width:7px; height:7px; background:#4caf7d; border-radius:50%;"></div> Online now</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        chat_container = st.container(height=360, border=False)
        with chat_container:
            if not st.session_state.chat_history:
                st.chat_message("assistant", avatar="🤖").write("Hi there! 👋 I'm the ShopFAQ support assistant. Ask me anything about **shipping**, **returns**, or **warranties**!")
            for msg in st.session_state.chat_history:
                st.chat_message("assistant" if msg["role"]=="bot" else "user", avatar="🤖" if msg["role"]=="bot" else "🙂").write(msg["content"])

        if prompt := st.chat_input("How can we help?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                # We don't need to write here manually because the rerun will pick it up
                # and display the history above. But for immediate feedback:
                with st.spinner(""):
                    res = app.invoke(
                        {"question": prompt},
                        config={"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    answer = res.get("answer", "I encountered an error. Please try again.")
                    st.session_state.chat_history.append({"role": "bot", "content": answer})
                    st.rerun()