import os
import datetime
from typing import TypedDict, List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables (API Keys) before anything else
load_dotenv()

# --- Expanded Knowledge Base (100-300 words each) ---
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

It is important to note that international customers are solely responsible for any customs duties, import taxes, or brokerage fees that may be levied by their country's customs department upon arrival. These fees are not included in our shipping rates and must be paid by the recipient. We provide all necessary documentation to ensure a smooth customs process, but we cannot be held responsible for delays caused by local customs inspections or regulatory holds. If a package is returned to us because of unpaid duties, the shipping cost will not be refunded. We recommend checking your local import laws before placing a high-value international order."""
    },
    {
        "id": "doc_003",
        "topic": "General Return Policy",
        "text": """We want you to be completely satisfied with your purchase. If for any reason you are not, you may return eligible items within 30 days of the date you received your order. To qualify for a refund, items must be in their original, pristine condition—this means they must be unwashed, unworn, and have all original tags and packaging intact. We cannot accept returns for items that show signs of wear or have been altered in any way.

To initiate a return, please visit our online Return Portal and enter your order number and email address. Once your return request is authorized, you will receive a prepaid shipping label (the cost of which will be deducted from your refund). Please note that 'Final Sale' items, personalized products, and certain hygiene-related goods are not eligible for return or exchange. A standard $5 restocking fee is applied to every return to cover the costs of inspection and processing. This fee will be automatically deducted from your final refund amount."""
    },
    {
        "id": "doc_004",
        "topic": "Refund Processing",
        "text": """Once your returned package reaches our warehouse, our quality assurance team will perform a thorough inspection to ensure the item meets our return criteria. This inspection process is typically completed within 2 business days of receipt. Following approval, we will initiate a refund to your original method of payment (e.g., the credit card or PayPal account used for the purchase).

While we process refunds immediately upon approval, please be aware that the time it takes for the credit to appear on your account depends on your financial institution. Generally, most customers see the refund reflected on their bank statement within 5 to 7 business days. You will receive an automated email notification once the refund has been successfully processed on our end. If you have not seen the credit after 10 business days, we recommend contacting your bank first, as they may have a pending transaction that hasn't posted to your visible balance yet."""
    },
    {
        "id": "doc_005",
        "topic": "Product Warranty",
        "text": """At our E-Commerce store, we stand behind the quality of our electronics. All electronic devices sold through our platform come with a 1-year limited manufacturer warranty. This warranty specifically covers any defects in materials or workmanship that occur under normal use during the first 12 months from the date of purchase. If your device malfunctions due to a manufacturing flaw, we will either repair it or provide a refurbished replacement at no cost to you.

Please be advised that this limited warranty does not cover damages resulting from accidents, liquid spills, improper handling, unauthorized repairs, or 'wear and tear' such as scratches or battery degradation. To file a warranty claim, you must provide your original order ID and a valid receipt. Our technical team may request photos or videos of the issue before authorizing a return for repair. The customer is responsible for shipping the item to our service center, while we cover the cost of shipping the repaired or replacement unit back to you."""
    },
    {
        "id": "doc_006",
        "topic": "Order Tracking",
        "text": """Staying informed about your package's journey is easy. As soon as your order is handed over to the carrier, we send you a 'Shipment Confirmation' email containing a unique tracking number and a direct link to the carrier's tracking page. You can also track your order at any time by logging into your account on our website, navigating to the 'My Orders' section, and clicking the 'Track Package' button next to your recent purchase.

Please keep in mind that tracking information may not be immediately available. It often takes up to 24 hours for the carrier to scan the package into their system and update the online status. If your tracking number shows 'Not Found' or 'Ready for Pickup' for more than two business days, please reach out to our support team. For high-value orders, a signature may be required upon delivery to ensure your items reach you safely."""
    },
    {
        "id": "doc_007",
        "topic": "Exchange Policy",
        "text": """To ensure you get the right item as quickly as possible, we do not offer direct exchanges. If you find that you need a different size, color, or style, the fastest method is to return your original item for a refund and place a new order for the desired product. This prevents delays caused by waiting for a return to arrive before a new item can be shipped.

Any promotional codes or 'Free Shipping' offers used on the original order cannot be transferred to a new replacement order unless the original item was received in a defective or damaged state. If you are returning an item and wish to re-order, we recommend placing the new order immediately while stock is available, rather than waiting for your refund to be processed. Our standard 30-day return policy applies to the original item, and the $5 restocking fee will still apply unless the item was faulty."""
    },
    {
        "id": "doc_008",
        "topic": "Missing or Damaged Items",
        "text": """We take great care in packaging your orders, but we understand that issues can occur during transit. If your package arrives damaged, or if you discover that an item is missing from your order, please contact our Customer Support team within 48 hours of the delivery timestamp. We require this prompt notification to file timely claims with our shipping carriers.

When reporting damage, please include your order number and clear photographs of the damaged item as well as the exterior and interior packaging. For missing items, please check the 'Shipment Confirmation' email to see if your order was split into multiple packages. If the item is truly missing, we will prioritize sending a replacement at no additional cost or issue a full refund for that specific item. Failure to report missing or damaged goods within the 48-hour window may result in the denial of your claim."""
    },
    {
        "id": "doc_009",
        "topic": "Customer Support Contacts",
        "text": """Our dedicated Customer Support team is here to help you with any questions or concerns you may have. We are available Monday through Friday, from 9:00 AM to 6:00 PM Eastern Standard Time (EST). We are closed on weekends and major public holidays, but you can leave us a message or send an email, and we will get back to you as soon as business resumes.

You can reach us through two main channels:
1. Email: Send your inquiries to support@shopfaq.example.com. We aim to respond to all emails within 24 hours during business days.
2. Phone: Call our toll-free support line at 1-800-SHOP-FAQ for immediate assistance with urgent issues.

When contacting us, please have your order number ready so we can assist you more efficiently. For technical issues with the website or your account, please describe the problem in detail or provide screenshots if possible."""
    },
    {
        "id": "doc_010",
        "topic": "Electronics Return Policy",
        "text": """Due to the sensitive nature of electronic hardware, we have a specialized return policy for these items that differs from our general policy. Electronics, including smartphones, laptops, and smart home devices, must be returned within 15 days of the delivery date. To be eligible for a full refund, the item must be completely unopened and still in its original, factory-sealed packaging.

If an electronic item has been opened, it is subject to a 15% 'Open-Box' restocking fee, which will be deducted from your refund. This fee reflects the reduced resale value of opened hardware. However, if the electronic item is found to be defective upon arrival, you may return it for a full refund or a replacement within the same 15-day window without any restocking fee. After the 15-day return period has passed, all defective electronics are covered by our 1-year limited manufacturer warranty rather than our return policy."""
    }
]

# --- State Design ---
class CapstoneState(TypedDict):
    question:      str
    messages:      List[dict]
    route:         str
    retrieved:     str
    sources:       List[str]
    tool_result:   str
    current_date:  str
    user_name:     str          # Added domain-specific field
    answer:        str
    faithfulness:  float
    eval_retries:  int
    language:      str          # Language for the response (e.g. "English", "Hindi", "French")

# --- Nodes ---

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    question = state["question"]
    
    # Extract name logic: "My name is [Name]"
    user_name = state.get("user_name", "")
    if "my name is" in question.lower():
        parts = question.lower().split("my name is")
        if len(parts) > 1:
            extracted = parts[1].strip().split()[0].capitalize()
            # Clean punctuation
            extracted = extracted.strip(".,!?")
            user_name = extracted

    msgs = msgs + [{"role": "user", "content": question}]
    if len(msgs) > 6:
        msgs = msgs[-6:]

    # Preserve existing language (set by UI); do not override it here
    language = state.get("language", "English")

    return {"messages": msgs, "user_name": user_name, "language": language}

# Router, Retrieval, Skip, Tool, Answer, Eval, Save nodes will be defined here...
# (I'll fill them with the logic from the notebook, but improved)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

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
    if "memory" in decision: decision = "memory_only"
    elif "tool" in decision: decision = "tool"
    else: decision = "retrieve"
    return {"route": decision}

def tool_node(state: CapstoneState) -> dict:
    today = datetime.date.today()
    return {"tool_result": f"The current date is {today}.", "current_date": str(today)}

def answer_node(state: CapstoneState) -> dict:
    question    = state["question"]
    retrieved   = state.get("retrieved", "")
    tool_result = state.get("tool_result", "")
    messages    = state.get("messages", [])
    eval_retries= state.get("eval_retries", 0)
    user_name   = state.get("user_name", "")
    language    = state.get("language", "English")

    context_parts = []
    if retrieved: context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
    if tool_result: context_parts.append(f"TOOL RESULT:\n{tool_result}")
    context = "\n\n".join(context_parts)

    system_prefix = (
        f"You are a customer support assistant. "
        f"IMPORTANT: You MUST respond ONLY in {language}, no matter what language the user writes in."
    )
    if user_name:
        system_prefix += f" The user's name is {user_name}."

    if context:
        system_content = f"""{system_prefix}
Answer using ONLY the provided context. If not in context, say you don't know and provide support email support@shopfaq.example.com.
Do NOT hallucinate.
{context}"""
    else:
        system_content = f"{system_prefix} Answer based on history. If they asked for their name, tell them."

    if eval_retries > 0:
        system_content += "\n\nSTRICT: Your last answer was inaccurate. Use ONLY context."

    lc_msgs = [SystemMessage(content=system_content)]
    for msg in messages[:-1]:
        lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

def eval_node(state: CapstoneState) -> dict:
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:800]
    retries  = state.get("eval_retries", 0)
    if not context: return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness 0.0-1.0. Does the answer use ONLY the context?
Context: {context}
Answer: {answer[:400]}
Reply with ONLY a number."""
    try:
        score = float(llm.invoke(prompt).content.strip().split()[0])
    except:
        score = 0.5
    return {"faithfulness": score, "eval_retries": retries + 1}

def save_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    return {"messages": msgs + [{"role": "assistant", "content": state["answer"]}]}

# --- Graph Assembly ---

def route_decision(state: CapstoneState) -> str:
    route = state.get("route", "retrieve")
    if route == "tool": return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"

def eval_decision(state: CapstoneState) -> str:
    if state.get("faithfulness", 1.0) >= 0.7 or state.get("eval_retries", 0) >= 2:
        return "save"
    return "answer"

def get_app(collection=None, embedder=None):
    # This helper injects collection/embedder into the node
    def retrieval_node_local(state: CapstoneState) -> dict:
        if not collection or not embedder: return {"retrieved": "", "sources": []}
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    builder = StateGraph(CapstoneState)
    builder.add_node("memory", memory_node)
    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieval_node_local)
    builder.add_node("skip", skip_retrieval_node)
    builder.add_node("tool", tool_node)
    builder.add_node("answer", answer_node)
    builder.add_node("eval", eval_node)
    builder.add_node("save", save_node)

    builder.set_entry_point("memory")
    builder.add_edge("memory", "router")
    builder.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    builder.add_edge("retrieve", "answer")
    builder.add_edge("skip", "answer")
    builder.add_edge("tool", "answer")
    builder.add_edge("answer", "eval")
    builder.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    builder.add_edge("save", END)

    return builder.compile(checkpointer=MemorySaver())