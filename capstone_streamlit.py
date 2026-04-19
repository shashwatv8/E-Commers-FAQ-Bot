import streamlit as st
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

from agent import get_app, DOCUMENTS, CapstoneState, SUPPORTED_LANGUAGES
from sentence_transformers import SentenceTransformer
import chromadb

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ShopFAQ — E-Commerce Store", layout="wide", page_icon="🛍️")

# ── CSS + HTML ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #0d0d0d !important;
    color: #f0ede6 !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"] { display: none !important; }

[data-testid="stAppViewBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* NAVBAR */
.navbar {
    width: 100%; background: #0d0d0d;
    border-bottom: 1px solid #2a2a2a;
    padding: 18px 60px;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
}
.navbar-logo { font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 900; color: #e8c97e; }
.navbar-links { display: flex; gap: 36px; font-size: 13px; font-weight: 500; color: #888; letter-spacing: 1px; text-transform: uppercase; }
.navbar-links span { cursor: pointer; transition: color 0.2s; }
.navbar-links span:hover { color: #f0ede6; }

/* HERO */
.hero { background: linear-gradient(135deg, #111 0%, #1a1a1a 50%, #0f0f0f 100%); padding: 100px 60px; display: flex; align-items: center; justify-content: space-between; min-height: 520px; position: relative; overflow: hidden; }
.hero::before { content: ''; position: absolute; top: -80px; right: -80px; width: 500px; height: 500px; background: radial-gradient(circle, rgba(232,201,126,0.08) 0%, transparent 70%); border-radius: 50%; }
.hero-text { max-width: 560px; z-index: 1; }
.hero-eyebrow { font-size: 11px; font-weight: 600; color: #e8c97e; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 20px; }
.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(42px, 5vw, 72px); font-weight: 900; line-height: 1.05; color: #f0ede6; margin-bottom: 24px; }
.hero-title span { color: #e8c97e; }
.hero-sub { font-size: 16px; line-height: 1.7; color: #888; margin-bottom: 40px; }
.hero-cta { display: inline-block; background: #e8c97e; color: #0d0d0d; font-size: 13px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; padding: 16px 36px; border-radius: 2px; cursor: pointer; border: 2px solid #e8c97e; transition: all 0.2s; }
.hero-cta:hover { background: transparent; color: #e8c97e; }
.hero-cta-ghost { display: inline-block; margin-left: 16px; border: 2px solid #333; color: #888; font-size: 13px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; padding: 16px 36px; border-radius: 2px; cursor: pointer; transition: all 0.2s; }
.hero-image-placeholder { width: 360px; height: 360px; border: 1px solid #2a2a2a; border-radius: 4px; background: linear-gradient(145deg, #181818, #141414); display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; font-size: 80px; }
.hero-image-placeholder p { font-size: 13px; color: #444; letter-spacing: 1px; text-transform: uppercase; }

/* MARQUEE */
.marquee-wrap { background: #e8c97e; padding: 12px 0; overflow: hidden; white-space: nowrap; }
.marquee-inner { display: inline-block; animation: marquee 28s linear infinite; font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #0d0d0d; }
@keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

/* CATEGORIES */
.section { padding: 80px 60px; }
.section-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 40px; border-bottom: 1px solid #1e1e1e; padding-bottom: 20px; }
.section-title { font-family: 'Playfair Display', serif; font-size: 32px; font-weight: 700; color: #f0ede6; }
.section-link { font-size: 12px; font-weight: 600; color: #e8c97e; letter-spacing: 1.5px; text-transform: uppercase; cursor: pointer; }
.categories-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.category-card { background: #141414; border: 1px solid #1e1e1e; border-radius: 3px; padding: 36px 24px; text-align: center; cursor: pointer; transition: all 0.25s; }
.category-card:hover { background: #1a1a1a; border-color: #e8c97e; transform: translateY(-2px); }
.category-icon { font-size: 36px; margin-bottom: 14px; }
.category-name { font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 700; color: #f0ede6; margin-bottom: 6px; }
.category-count { font-size: 12px; color: #555; }

/* PRODUCTS */
.products-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
.product-card { background: #111; border: 1px solid #1e1e1e; border-radius: 3px; overflow: hidden; cursor: pointer; transition: all 0.25s; }
.product-card:hover { border-color: #333; transform: translateY(-3px); box-shadow: 0 20px 40px rgba(0,0,0,0.5); }
.product-image { height: 200px; background: #181818; display: flex; align-items: center; justify-content: center; font-size: 56px; border-bottom: 1px solid #1e1e1e; position: relative; }
.product-badge { position: absolute; top: 14px; left: 14px; background: #e8c97e; color: #0d0d0d; font-size: 9px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; padding: 4px 10px; border-radius: 1px; }
.product-info { padding: 20px; }
.product-category { font-size: 10px; font-weight: 600; color: #555; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
.product-name { font-family: 'Playfair Display', serif; font-size: 18px; font-weight: 700; color: #f0ede6; margin-bottom: 12px; line-height: 1.3; }
.product-footer { display: flex; align-items: center; justify-content: space-between; }
.product-price { font-size: 20px; font-weight: 700; color: #e8c97e; }
.product-old-price { font-size: 13px; color: #444; text-decoration: line-through; margin-left: 8px; }
.add-to-cart { background: transparent; border: 1px solid #2a2a2a; color: #888; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; padding: 8px 16px; border-radius: 2px; cursor: pointer; transition: all 0.2s; }
.add-to-cart:hover { border-color: #e8c97e; color: #e8c97e; }

/* PROMO */
.promo-strip { background: #111; border-top: 1px solid #1e1e1e; border-bottom: 1px solid #1e1e1e; padding: 50px 60px; display: grid; grid-template-columns: repeat(3, 1fr); }
.promo-item { padding: 20px 40px; text-align: center; border-right: 1px solid #1e1e1e; }
.promo-item:last-child { border-right: none; }
.promo-icon { font-size: 28px; margin-bottom: 12px; }
.promo-title { font-family: 'Playfair Display', serif; font-size: 17px; font-weight: 700; color: #f0ede6; margin-bottom: 6px; }
.promo-desc { font-size: 13px; color: #555; line-height: 1.5; }

/* FOOTER */
.footer { background: #080808; border-top: 1px solid #1a1a1a; padding: 60px 60px 30px; }
.footer-grid { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 40px; margin-bottom: 50px; }
.footer-brand { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 900; color: #e8c97e; margin-bottom: 14px; }
.footer-desc { font-size: 13px; color: #444; line-height: 1.7; max-width: 240px; }
.footer-col-title { font-size: 11px; font-weight: 700; color: #f0ede6; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 18px; }
.footer-links { display: flex; flex-direction: column; gap: 10px; }
.footer-links a { font-size: 13px; color: #444; text-decoration: none; cursor: pointer; transition: color 0.2s; }
.footer-links a:hover { color: #f0ede6; }
.footer-bottom { border-top: 1px solid #1a1a1a; padding-top: 24px; display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: #333; }

/* CHAT BUBBLE */
#chat-toggle { position: fixed; bottom: 32px; right: 32px; width: 60px; height: 60px; background: #e8c97e; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 26px; cursor: pointer; z-index: 9998; box-shadow: 0 8px 32px rgba(232,201,126,0.3); transition: all 0.25s; border: none; animation: pulse 3s ease-in-out infinite; }
#chat-toggle:hover { transform: scale(1.1); }
@keyframes pulse { 0%,100%{box-shadow:0 8px 32px rgba(232,201,126,0.3);}50%{box-shadow:0 8px 48px rgba(232,201,126,0.5);} }
.chat-badge { position: absolute; top: -2px; right: -2px; width: 18px; height: 18px; background: #ff4444; border-radius: 50%; border: 2px solid #0d0d0d; display: flex; align-items: center; justify-content: center; font-size: 9px; font-weight: 800; color: white; }

/* CHAT WINDOW */
#chat-window { position: fixed; bottom: 106px; right: 32px; width: 390px; height: 560px; background: #111; border: 1px solid #2a2a2a; border-radius: 12px; z-index: 9999; display: flex; flex-direction: column; box-shadow: 0 24px 80px rgba(0,0,0,0.8); transform: scale(0.92) translateY(20px); opacity: 0; pointer-events: none; transition: all 0.28s cubic-bezier(0.34,1.56,0.64,1); overflow: hidden; }
#chat-window.open { transform: scale(1) translateY(0); opacity: 1; pointer-events: all; }
.chat-header { background: #0d0d0d; border-bottom: 1px solid #1e1e1e; padding: 14px 18px; display: flex; align-items: center; justify-content: space-between; }
.chat-header-left { display: flex; align-items: center; gap: 12px; }
.chat-avatar { width: 38px; height: 38px; background: #e8c97e; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
.chat-name { font-family: 'Playfair Display', serif; font-size: 15px; font-weight: 700; color: #f0ede6; }
.chat-status { font-size: 11px; color: #4caf7d; display: flex; align-items: center; gap: 5px; }
.status-dot { width: 7px; height: 7px; background: #4caf7d; border-radius: 50%; }
.chat-close { background: none; border: none; color: #444; font-size: 20px; cursor: pointer; transition: color 0.2s; }
.chat-close:hover { color: #f0ede6; }

/* Language badge in header */
.lang-badge { background: #1e1e1e; border: 1px solid #333; border-radius: 20px; padding: 3px 10px; font-size: 11px; color: #e8c97e; font-weight: 600; margin-left: 8px; }

.chat-messages { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px; scrollbar-width: thin; scrollbar-color: #2a2a2a transparent; }
.chat-messages::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 2px; }
.msg { display: flex; gap: 8px; align-items: flex-end; max-width: 90%; }
.msg.user { flex-direction: row-reverse; align-self: flex-end; }
.msg.bot  { align-self: flex-start; }
.msg-bubble { padding: 10px 14px; border-radius: 12px; font-size: 13.5px; line-height: 1.55; word-break: break-word; }
.msg.user .msg-bubble { background: #e8c97e; color: #0d0d0d; font-weight: 500; border-bottom-right-radius: 3px; }
.msg.bot  .msg-bubble { background: #1a1a1a; color: #d0cdc6; border: 1px solid #2a2a2a; border-bottom-left-radius: 3px; }
.msg-avatar { width: 26px; height: 26px; border-radius: 50%; background: #1e1e1e; border: 1px solid #2a2a2a; display: flex; align-items: center; justify-content: center; font-size: 13px; flex-shrink: 0; }
.typing { display: flex; align-items: center; gap: 4px; padding: 10px 14px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 12px; border-bottom-left-radius: 3px; }
.typing span { width: 6px; height: 6px; background: #555; border-radius: 50%; animation: typing 1.2s ease-in-out infinite; }
.typing span:nth-child(2){animation-delay:0.2s;} .typing span:nth-child(3){animation-delay:0.4s;}
@keyframes typing { 0%,60%,100%{transform:translateY(0);background:#555;}30%{transform:translateY(-6px);background:#e8c97e;} }

.chat-topics { padding: 8px 14px 4px; display: flex; flex-wrap: wrap; gap: 6px; border-bottom: 1px solid #1a1a1a; }
.topic-chip { background: #1a1a1a; border: 1px solid #2a2a2a; color: #888; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; padding: 4px 10px; border-radius: 20px; cursor: pointer; transition: all 0.2s; }
.topic-chip:hover { border-color: #e8c97e; color: #e8c97e; }

.chat-input-area { padding: 12px 14px; background: #0d0d0d; border-top: 1px solid #1e1e1e; display: flex; gap: 10px; align-items: center; }
.chat-input-area input { flex: 1; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 10px 14px; color: #f0ede6; font-size: 13px; font-family: 'DM Sans', sans-serif; outline: none; transition: border-color 0.2s; }
.chat-input-area input:focus { border-color: #e8c97e; }
.chat-input-area input::placeholder { color: #444; }
.chat-send { width: 38px; height: 38px; background: #e8c97e; border: none; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 16px; transition: all 0.2s; flex-shrink: 0; }
.chat-send:hover { background: #f5d98a; transform: scale(1.05); }
.chat-send:disabled { background: #2a2a2a; cursor: not-allowed; transform: none; }

/* Language hint bar */
.lang-hint { padding: 5px 14px 0; text-align: center; }
.lang-hint span { font-size: 11px; color: #555; }
</style>

<!-- NAVBAR -->
<div class="navbar">
  <div class="navbar-logo">🛍️ ShopFAQ</div>
  <div class="navbar-links">
    <span>New Arrivals</span><span>Electronics</span><span>Apparel</span>
    <span>Home & Living</span><span>Sale</span>
  </div>
  <div style="display:flex;gap:20px;align-items:center;">
    <span style="font-size:13px;color:#888;cursor:pointer;">🔍</span>
    <span style="font-size:13px;color:#888;cursor:pointer;">👤</span>
    <span style="font-size:20px;cursor:pointer;position:relative;">🛒<span style="position:absolute;top:-6px;right:-8px;background:#e8c97e;color:#0d0d0d;font-size:9px;font-weight:700;border-radius:50%;width:16px;height:16px;display:flex;align-items:center;justify-content:center;">3</span></span>
  </div>
</div>

<!-- HERO -->
<div class="hero">
  <div class="hero-text">
    <div class="hero-eyebrow">New Season Collection</div>
    <h1 class="hero-title">Premium Products,<br><span>Exceptional</span><br>Service.</h1>
    <p class="hero-sub">Discover curated electronics, fashion, and home essentials — shipped fast, backed by our no-fuss return policy. Ask our AI assistant in <strong style="color:#e8c97e;">any language</strong>.</p>
    <a class="hero-cta">Shop Now</a>
    <a class="hero-cta-ghost">Explore Deals</a>
  </div>
  <div class="hero-image-placeholder">🛍️<p>New Collection 2025</p></div>
</div>

<!-- MARQUEE -->
<div class="marquee-wrap">
  <div class="marquee-inner">
    &nbsp;&nbsp;FREE SHIPPING ON ORDERS OVER $50 &nbsp;•&nbsp; 30-DAY RETURNS &nbsp;•&nbsp; 1-YEAR WARRANTY &nbsp;•&nbsp; SHIPS TO 50+ COUNTRIES &nbsp;•&nbsp; AI SUPPORT IN 12+ LANGUAGES &nbsp;•&nbsp; FREE SHIPPING ON ORDERS OVER $50 &nbsp;•&nbsp; 30-DAY RETURNS &nbsp;•&nbsp; 1-YEAR WARRANTY &nbsp;•&nbsp; SHIPS TO 50+ COUNTRIES &nbsp;•&nbsp; AI SUPPORT IN 12+ LANGUAGES &nbsp;&nbsp;
  </div>
</div>

<!-- CATEGORIES -->
<div class="section">
  <div class="section-header"><div class="section-title">Browse Categories</div><div class="section-link">View All →</div></div>
  <div class="categories-grid">
    <div class="category-card"><div class="category-icon">💻</div><div class="category-name">Electronics</div><div class="category-count">240+ items</div></div>
    <div class="category-card"><div class="category-icon">👗</div><div class="category-name">Apparel</div><div class="category-count">580+ items</div></div>
    <div class="category-card"><div class="category-icon">🏠</div><div class="category-name">Home & Living</div><div class="category-count">320+ items</div></div>
    <div class="category-card"><div class="category-icon">🎧</div><div class="category-name">Accessories</div><div class="category-count">190+ items</div></div>
  </div>
</div>

<!-- PRODUCTS -->
<div class="section" style="padding-top:0;">
  <div class="section-header"><div class="section-title">Featured Products</div><div class="section-link">Shop All →</div></div>
  <div class="products-grid">
    <div class="product-card">
      <div class="product-image">💻<div class="product-badge">Best Seller</div></div>
      <div class="product-info"><div class="product-category">Electronics</div><div class="product-name">UltraBook Pro 15"</div>
        <div class="product-footer"><div><span class="product-price">$899</span><span class="product-old-price">$1,099</span></div><button class="add-to-cart">+ Add</button></div></div>
    </div>
    <div class="product-card">
      <div class="product-image">🎧<div class="product-badge">New</div></div>
      <div class="product-info"><div class="product-category">Accessories</div><div class="product-name">NoiseShield Pro Headphones</div>
        <div class="product-footer"><div><span class="product-price">$199</span><span class="product-old-price">$249</span></div><button class="add-to-cart">+ Add</button></div></div>
    </div>
    <div class="product-card">
      <div class="product-image">📱</div>
      <div class="product-info"><div class="product-category">Electronics</div><div class="product-name">SmartPhone X12 Ultra</div>
        <div class="product-footer"><div><span class="product-price">$649</span></div><button class="add-to-cart">+ Add</button></div></div>
    </div>
  </div>
</div>

<!-- PROMO -->
<div class="promo-strip">
  <div class="promo-item"><div class="promo-icon">🚚</div><div class="promo-title">Free Shipping</div><div class="promo-desc">On all orders above $50. Standard 3–5 day delivery.</div></div>
  <div class="promo-item"><div class="promo-icon">↩️</div><div class="promo-title">Easy Returns</div><div class="promo-desc">30-day no-fuss returns. We handle the label.</div></div>
  <div class="promo-item"><div class="promo-icon">🌐</div><div class="promo-title">12+ Languages</div><div class="promo-desc">Our AI assistant speaks your language — literally.</div></div>
</div>

<!-- FOOTER -->
<div class="footer">
  <div class="footer-grid">
    <div><div class="footer-brand">ShopFAQ</div><div class="footer-desc">Your destination for premium products with AI-powered multilingual support.</div></div>
    <div><div class="footer-col-title">Support</div><div class="footer-links"><a>Track Order</a><a>Return Portal</a><a>Warranty Claims</a><a>Contact Us</a></div></div>
    <div><div class="footer-col-title">Company</div><div class="footer-links"><a>About</a><a>Careers</a><a>Blog</a><a>Press</a></div></div>
    <div><div class="footer-col-title">Legal</div><div class="footer-links"><a>Privacy Policy</a><a>Terms of Service</a><a>Cookie Policy</a></div></div>
  </div>
  <div class="footer-bottom"><span>© 2025 ShopFAQ. All rights reserved.</span><span>support@shopfaq.example.com &nbsp;|&nbsp; 1-800-SHOP-FAQ</span></div>
</div>

<!-- CHAT BUBBLE -->
<button id="chat-toggle" onclick="toggleChat()" title="Chat with Support">
  <span id="chat-icon">💬</span>
  <div class="chat-badge" id="chat-notif">1</div>
</button>

<!-- CHAT WINDOW -->
<div id="chat-window">
  <div class="chat-header">
    <div class="chat-header-left">
      <div class="chat-avatar">🤖</div>
      <div>
        <div style="display:flex;align-items:center;">
          <div class="chat-name">ShopFAQ Assistant</div>
          <div class="lang-badge" id="lang-badge">EN</div>
        </div>
        <div class="chat-status"><div class="status-dot"></div> Supports 12+ languages</div>
      </div>
    </div>
    <button class="chat-close" onclick="toggleChat()">✕</button>
  </div>

  <div class="lang-hint">
    <span>💬 Type in any language — Hindi, Telugu, Tamil, French, Spanish, Arabic &amp; more</span>
  </div>

  <div class="chat-topics" id="chat-topics">
    <div class="topic-chip" onclick="sendQuick('What are your shipping options?')">Shipping</div>
    <div class="topic-chip" onclick="sendQuick('How do I return an item?')">Returns</div>
    <div class="topic-chip" onclick="sendQuick('What is the warranty on electronics?')">Warranty</div>
    <div class="topic-chip" onclick="sendQuick('शिपिंग पॉलिसी क्या है?')">हिंदी</div>
    <div class="topic-chip" onclick="sendQuick('రిటర్న్ పాలసీ ఏమిటి?')">తెలుగు</div>
    <div class="topic-chip" onclick="sendQuick('Quelle est votre politique de retour?')">Français</div>
  </div>

  <div class="chat-messages" id="chat-messages">
    <div class="msg bot">
      <div class="msg-avatar">🤖</div>
      <div class="msg-bubble">
        Hi! 👋 I'm the ShopFAQ assistant. Ask me anything about <strong>shipping</strong>, <strong>returns</strong>, or <strong>warranties</strong> — in <strong>any language</strong>!<br><br>
        <span style="font-size:12px;color:#888;">हिंदी • తెలుగు • தமிழ் • Français • Español • العربية • 中文 • and more</span>
      </div>
    </div>
  </div>

  <div class="chat-input-area">
    <input type="text" id="chat-input" placeholder="Ask in any language…" onkeydown="handleKey(event)" />
    <button class="chat-send" id="send-btn" onclick="sendMessage()">➤</button>
  </div>
</div>

<script>
let chatOpen = false;

// Language code → flag + short label
const LANG_LABELS = {
  en:"🇬🇧 EN", hi:"🇮🇳 HI", te:"🇮🇳 TE", ta:"🇮🇳 TA", bn:"🇮🇳 BN",
  mr:"🇮🇳 MR", fr:"🇫🇷 FR", es:"🇪🇸 ES", de:"🇩🇪 DE",
  ar:"🇸🇦 AR", zh:"🇨🇳 ZH", ja:"🇯🇵 JA"
};

function toggleChat() {
  chatOpen = !chatOpen;
  const win = document.getElementById('chat-window');
  const icon = document.getElementById('chat-icon');
  const notif = document.getElementById('chat-notif');
  if (chatOpen) { win.classList.add('open'); icon.textContent = '✕'; notif.style.display='none'; }
  else          { win.classList.remove('open'); icon.textContent = '💬'; }
}
function handleKey(e) { if (e.key === 'Enter') sendMessage(); }
function sendQuick(text) { document.getElementById('chat-input').value = text; sendMessage(); }

function appendMessage(role, text) {
  const msgs = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const escaped = text.replace(/</g,'&lt;').replace(/>/g,'&gt;');
  if (role === 'bot') {
    div.innerHTML = '<div class="msg-avatar">🤖</div><div class="msg-bubble">' + escaped + '</div>';
  } else {
    div.innerHTML = '<div class="msg-bubble">' + escaped + '</div><div class="msg-avatar">🙂</div>';
  }
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function showTyping() {
  const msgs = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className='msg bot'; div.id='typing-indicator';
  div.innerHTML='<div class="msg-avatar">🤖</div><div class="typing"><span></span><span></span><span></span></div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}
function hideTyping() { const t=document.getElementById('typing-indicator'); if(t) t.remove(); }

function updateLangBadge(code) {
  const badge = document.getElementById('lang-badge');
  badge.textContent = LANG_LABELS[code] || code.toUpperCase();
}

function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  appendMessage('user', text);
  document.getElementById('send-btn').disabled = true;
  showTyping();
  window.parent.postMessage({type:'chatMessage', text}, '*');
  setTimeout(() => { hideTyping(); document.getElementById('send-btn').disabled = false; }, 600);
}

window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'botReply') {
    hideTyping();
    appendMessage('bot', e.data.text);
    document.getElementById('send-btn').disabled = false;
    if (e.data.lang) updateLangBadge(e.data.lang);
  }
});
</script>
""", unsafe_allow_html=True)

# ── Backend ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_agent_and_kb():
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    client     = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")
    texts      = [d["text"] for d in DOCUMENTS]
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

if "thread_id"  not in st.session_state: st.session_state.thread_id  = str(uuid.uuid4())[:8]
if "last_msg"   not in st.session_state: st.session_state.last_msg   = ""

# Hidden relay input (invisible to user)
st.markdown("""
<style>
div[data-testid="stTextInput"] {
    position: fixed !important; bottom: -300px !important;
    opacity: 0 !important; pointer-events: none !important; height: 0 !important;
}
</style>""", unsafe_allow_html=True)

user_msg = st.text_input("relay", key="chat_relay", label_visibility="hidden")

if user_msg and user_msg != st.session_state.last_msg:
    st.session_state.last_msg = user_msg
    with st.spinner(""):
        res    = app.invoke(
            {"question": user_msg},
            config={"configurable": {"thread_id": st.session_state.thread_id}}
        )
        answer = res.get("answer", "Sorry, something went wrong. Please try again.")
        lang   = res.get("detected_lang", "en")

    # Send reply + detected language back to the JS chat window
    safe_answer = (answer
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("'", "\\'")
        .replace("\n", "<br>"))
    st.markdown(f"""
    <script>
    (function(){{
        window.parent.postMessage({{type:'botReply', text:'{safe_answer}', lang:'{lang}'}}, '*');
    }})();
    </script>""", unsafe_allow_html=True)