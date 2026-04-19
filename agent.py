import os
import datetime
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Knowledge Base (10 documents — English source, answers generated in user's language)
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Standard & Express Shipping",
        "text": """We offer two primary shipping tiers for domestic orders: Standard and Express. Standard shipping is our most economical option, costing a flat rate of $5.99 for all orders under $50. Once your order is processed, Standard shipping typically delivers within 3 to 5 business days, depending on your distance from our fulfillment centers in Hyderabad and Bangalore. For customers who need their items sooner, we offer Express shipping for a flat rate of $14.99. Express orders are prioritized in our warehouse and delivered within 1 to 2 business days.

A key benefit for our frequent shoppers is our Free Standard Shipping policy, which applies automatically to all domestic orders with a subtotal of $50 or more after all discounts have been applied. Please note that order processing takes up to 24 hours. Orders placed after 2:00 PM EST on weekdays, or any time during the weekend, will begin processing on the next business day. You will receive a notification with tracking details as soon as your package leaves our facility."""
    },
    {
        "id": "doc_002",
        "topic": "International Shipping",
        "text": """Our store is proud to offer international shipping to over 50 countries worldwide, including major destinations in North America, Europe, and Asia-Pacific. International orders are shipped via our global logistics partners and generally arrive within 7 to 14 business days. We charge a flat international shipping fee of $25 per order, regardless of the package weight or destination.

It is important to note that international customers are solely responsible for any customs duties, import taxes, or brokerage fees that may be levied by their country's customs department upon arrival. These fees are not included in our shipping rates and must be paid by the recipient. We provide all necessary documentation to ensure a smooth customs process, but we cannot be held responsible for delays caused by local customs inspections or regulatory holds. If a package is returned to us because of unpaid duties, the shipping cost will not be refunded."""
    },
    {
        "id": "doc_003",
        "topic": "General Return Policy",
        "text": """We want you to be completely satisfied with your purchase. If for any reason you are not, you may return eligible items within 30 days of the date you received your order. To qualify for a refund, items must be in their original, pristine condition — unwashed, unworn, and with all original tags and packaging intact. We cannot accept returns for items that show signs of wear or have been altered in any way.

To initiate a return, please visit our online Return Portal and enter your order number and email address. Once your return request is authorized, you will receive a prepaid shipping label (the cost of which will be deducted from your refund). Final Sale items, personalized products, and certain hygiene-related goods are not eligible for return or exchange. A standard $5 restocking fee is applied to every return."""
    },
    {
        "id": "doc_004",
        "topic": "Refund Processing",
        "text": """Once your returned package reaches our warehouse, our quality assurance team will perform a thorough inspection within 2 business days of receipt. Following approval, we will initiate a refund to your original method of payment. Generally, most customers see the refund reflected on their bank statement within 5 to 7 business days. You will receive an automated email notification once the refund has been successfully processed. If you have not seen the credit after 10 business days, we recommend contacting your bank first."""
    },
    {
        "id": "doc_005",
        "topic": "Product Warranty",
        "text": """All electronic devices sold through our platform come with a 1-year limited manufacturer warranty. This warranty covers defects in materials or workmanship that occur under normal use during the first 12 months from the date of purchase. If your device malfunctions due to a manufacturing flaw, we will either repair it or provide a refurbished replacement at no cost to you.

This warranty does not cover damages from accidents, liquid spills, improper handling, unauthorized repairs, or wear and tear such as scratches or battery degradation. To file a warranty claim, provide your original order ID and a valid receipt. The customer is responsible for shipping the item to our service center; we cover the return shipping cost."""
    },
    {
        "id": "doc_006",
        "topic": "Order Tracking",
        "text": """As soon as your order is handed over to the carrier, we send you a Shipment Confirmation email containing a unique tracking number and a direct link to the carrier's tracking page. You can also track your order by logging into your account and navigating to My Orders. Tracking information may not be immediately available — it often takes up to 24 hours for the carrier to scan the package. If your tracking number shows Not Found for more than two business days, please reach out to our support team."""
    },
    {
        "id": "doc_007",
        "topic": "Exchange Policy",
        "text": """We do not offer direct exchanges. If you need a different size, color, or style, the fastest method is to return your original item for a refund and place a new order for the desired product. Promotional codes or Free Shipping offers used on the original order cannot be transferred to a replacement order unless the item was received defective or damaged. Our standard 30-day return policy applies to the original item, and the $5 restocking fee will still apply unless the item was faulty."""
    },
    {
        "id": "doc_008",
        "topic": "Missing or Damaged Items",
        "text": """If your package arrives damaged or an item is missing, please contact our Customer Support team within 48 hours of the delivery timestamp. When reporting damage, include your order number and clear photographs of the damaged item and its packaging. For missing items, first check your Shipment Confirmation email to see if your order was split into multiple packages. If the item is truly missing, we will send a replacement at no additional cost or issue a full refund. Failure to report within 48 hours may result in denial of your claim."""
    },
    {
        "id": "doc_009",
        "topic": "Customer Support Contacts",
        "text": """Our Customer Support team is available Monday through Friday, 9:00 AM to 6:00 PM EST. We are closed on weekends and major public holidays, but you can email us and we will respond on the next business day.

Contact channels:
1. Email: support@shopfaq.example.com — we aim to respond within 24 hours on business days.
2. Phone: 1-800-SHOP-FAQ — for immediate assistance with urgent issues.

Please have your order number ready when contacting us so we can assist you more efficiently."""
    },
    {
        "id": "doc_010",
        "topic": "Electronics Return Policy",
        "text": """Electronics, including smartphones, laptops, and smart home devices, must be returned within 15 days of the delivery date. To be eligible for a full refund, the item must be completely unopened and still in its original, factory-sealed packaging. If an electronic item has been opened, it is subject to a 15% Open-Box restocking fee. If the electronic item is defective upon arrival, you may return it for a full refund or replacement within the same 15-day window with no restocking fee. After 15 days, defective electronics are covered by our 1-year limited manufacturer warranty."""
    }
]

# ---------------------------------------------------------------------------
# Supported languages for display in the UI
# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "te": "Telugu (తెలుగు)",
    "ta": "Tamil (தமிழ்)",
    "bn": "Bengali (বাংলা)",
    "mr": "Marathi (मराठी)",
    "fr": "French (Français)",
    "es": "Spanish (Español)",
    "de": "German (Deutsch)",
    "ar": "Arabic (العربية)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
}

# ---------------------------------------------------------------------------
# State — added language detection fields
# ---------------------------------------------------------------------------
class CapstoneState(TypedDict):
    question:        str
    messages:        List[dict]
    route:           str
    retrieved:       str
    sources:         List[str]
    tool_result:     str
    current_date:    str
    user_name:       str
    answer:          str
    faithfulness:    float
    eval_retries:    int
    detected_lang:   str   # ISO 639-1 code detected from the user's message
    language_name:   str   # Human-readable name, e.g. "Hindi"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def memory_node(state: CapstoneState) -> dict:
    msgs     = state.get("messages", [])
    question = state["question"]

    # Extract user name from "my name is ..." (works across languages via English pattern)
    user_name = state.get("user_name", "")
    if "my name is" in question.lower():
        parts = question.lower().split("my name is")
        if len(parts) > 1:
            extracted = parts[1].strip().split()[0].capitalize().strip(".,!?")
            user_name = extracted

    msgs = msgs + [{"role": "user", "content": question}]
    if len(msgs) > 6:
        msgs = msgs[-6:]

    return {"messages": msgs, "user_name": user_name}


def language_detection_node(state: CapstoneState) -> dict:
    """
    Detects the language of the user's question using the LLM.
    Returns the ISO 639-1 code and a human-readable name.
    If the user has previously established a language in this session,
    that language is preserved unless a clear switch is detected.
    """
    question      = state["question"]
    prev_lang     = state.get("detected_lang", "")

    prompt = f"""Detect the language of the following text and return ONLY the ISO 639-1 two-letter language code (e.g. en, hi, te, ta, fr, es, de, ar, zh, ja, bn, mr).
Do NOT include any explanation or punctuation — just the code.

Text: {question}

Language code:"""

    try:
        code = llm.invoke(prompt).content.strip().lower()[:5].split()[0]
        code = ''.join(c for c in code if c.isalpha())[:2]
    except Exception:
        code = prev_lang or "en"

    # Fall back to previous detected language if detection is ambiguous
    if len(code) != 2:
        code = prev_lang or "en"

    language_name = SUPPORTED_LANGUAGES.get(code, f"Language ({code})")
    return {"detected_lang": code, "language_name": language_name}


def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(
        f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]
    ) or "none"

    prompt = f"""You are a router for an E-Commerce FAQ chatbot.
Available options:
- retrieve: search the KB for shipping, returns, warranty, and order inquiries.
- memory_only: answer from history (greetings, 'what is my name?', 'what did I just say?')
- tool: use 'current_date' if the user asks for today's date or relative time.

Recent conversation: {recent}
Current question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""

    response = llm.invoke(prompt)
    decision = response.content.strip().lower()
    if "memory"  in decision: decision = "memory_only"
    elif "tool"  in decision: decision = "tool"
    else:                     decision = "retrieve"
    return {"route": decision}


def tool_node(state: CapstoneState) -> dict:
    today = datetime.date.today()
    return {"tool_result": f"The current date is {today}.", "current_date": str(today)}


def answer_node(state: CapstoneState) -> dict:
    question      = state["question"]
    retrieved     = state.get("retrieved", "")
    tool_result   = state.get("tool_result", "")
    messages      = state.get("messages", [])
    eval_retries  = state.get("eval_retries", 0)
    user_name     = state.get("user_name", "")
    detected_lang = state.get("detected_lang", "en")
    language_name = state.get("language_name", "English")

    context_parts = []
    if retrieved:   context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result: context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    system_prefix = "You are a helpful e-commerce customer support assistant."
    if user_name:
        system_prefix += f" The user's name is {user_name}."

    # ── Language instruction ──────────────────────────────────────────────────
    if detected_lang == "en":
        lang_instruction = "Respond in English."
    else:
        lang_instruction = (
            f"IMPORTANT: The user wrote in {language_name}. "
            f"You MUST respond entirely in {language_name}. "
            f"Do NOT reply in English unless the user explicitly asks for English."
        )

    if context:
        system_content = f"""{system_prefix}
{lang_instruction}
Answer using ONLY the provided context. If the answer is not in the context, say you don't know and provide the support email support@shopfaq.example.com.
Do NOT hallucinate or invent information.
{context}"""
    else:
        system_content = (
            f"{system_prefix}\n{lang_instruction}\n"
            "Answer based on conversation history only. If they asked for their name, tell them."
        )

    if eval_retries > 0:
        system_content += "\n\nSTRICT: Your last answer was inaccurate. Use ONLY the provided context."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        role = msg["role"]
        lc_msgs.append(
            HumanMessage(content=msg["content"]) if role == "user"
            else AIMessage(content=msg["content"])
        )
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}


def eval_node(state: CapstoneState) -> dict:
    answer  = state.get("answer", "")
    context = state.get("retrieved", "")[:800]
    retries = state.get("eval_retries", 0)

    if not context:
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness 0.0-1.0. Does the answer use ONLY the context?
Context: {context}
Answer: {answer[:400]}
Reply with ONLY a number."""
    try:
        score = float(llm.invoke(prompt).content.strip().split()[0])
    except Exception:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}


def save_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    return {"messages": msgs + [{"role": "assistant", "content": state["answer"]}]}

# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    if state.get("faithfulness", 1.0) >= 0.7 or state.get("eval_retries", 0) >= 2:
        return "save"
    return "answer"


def get_app(collection=None, embedder=None):

    def retrieval_node_local(state: CapstoneState) -> dict:
        if not collection or not embedder:
            return {"retrieved": "", "sources": []}
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(
            f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    builder = StateGraph(CapstoneState)

    # Register all nodes
    builder.add_node("memory",   memory_node)
    builder.add_node("lang",     language_detection_node)   
    builder.add_node("router",   router_node)
    builder.add_node("retrieve", retrieval_node_local)
    builder.add_node("skip",     skip_retrieval_node)
    builder.add_node("tool",     tool_node)
    builder.add_node("answer",   answer_node)
    builder.add_node("eval",     eval_node)
    builder.add_node("save",     save_node)

    # Wire edges
    # memory → lang → router (lang detection happens before routing)
    builder.set_entry_point("memory")
    builder.add_edge("memory",   "lang")
    builder.add_edge("lang",     "router")
    builder.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )
    builder.add_edge("retrieve", "answer")
    builder.add_edge("skip",     "answer")
    builder.add_edge("tool",     "answer")
    builder.add_edge("answer",   "eval")
    builder.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    builder.add_edge("save", END)

    return builder.compile(checkpointer=MemorySaver())