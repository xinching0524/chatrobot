import streamlit as st
import os
import base64
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pypdf import PdfReader
from PIL import Image
import io

# 加載環境變數
load_dotenv()

# --- [新增功能] JSON 存檔邏輯 ---
def save_chat_to_json(messages):
    chat_history_to_save = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue  # 不存系統提示
            
        role = "user" if isinstance(msg, HumanMessage) else "ai"
        
        # 處理多模態內容 (圖片/文字混和)
        content_text = ""
        if isinstance(msg.content, list):
            for item in msg.content:
                if item["type"] == "text":
                    content_text += item["text"]
        else:
            content_text = msg.content

        chat_history_to_save.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content_text
        })
    
    if chat_history_to_save:
        # 檔名格式: chat_YYYYMMDD_HHMMSS.json
        filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chat_history_to_save, f, ensure_ascii=False, indent=4)
        return filename
    return None

# 網頁配置
st.set_page_config(page_title="Gemini 繁體中文 AI 助手", page_icon="🤖", layout="wide")

# 自定義 CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stChatMessage { border-radius: 15px; padding: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="你是 Gemini，由 Google 開發。請始終使用『繁體中文(台灣)』回答，語氣友善專業。")
    ]

# 側邊欄
with st.sidebar:
    st.title("⚙️ 設定與檔案")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        api_key = st.text_input("輸入 Google API Key:", type="password")
    
    selected_model = st.selectbox("選擇模型:", ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"])
    
    st.divider()
    
    # --- [新增功能] 存檔按鈕 ---
    if st.button("💾 結束對話並存檔 (JSON)"):
        saved_file = save_chat_to_json(st.session_state.messages)
        if saved_file:
            st.success(f"存檔成功！檔名：{saved_file}")
        else:
            st.warning("目前沒有可儲存的對話。")
            
    if st.button("🗑️ 清除對話紀錄"):
        st.session_state.messages = [
            SystemMessage(content="你是 Gemini，由 Google 開發。請始終使用『繁體中文(台灣)』回答，語氣友善專業。")
        ]
        st.rerun()

    st.divider()
    st.subheader("📁 上傳檔案分析")
    uploaded_file = st.file_uploader("支援圖片 (JPG/PNG), PDF 或 TXT", type=["jpg", "jpeg", "png", "pdf", "txt"])

st.title("🤖 Gemini 繁體中文 AI 助手")
st.caption("基於 LangChain 與 Google Gemini API 打造的網頁版對話機器人")

# 顯示聊天歷史
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            if isinstance(msg.content, list):
                for item in msg.content:
                    if item["type"] == "text": st.markdown(item["text"])
                    elif item["type"] == "image_url": st.image(item["image_url"])
            else:
                st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 聊天輸入框
if prompt := st.chat_input("輸入訊息..."):
    if not api_key:
        st.error("請提供 API Key 以繼續。")
        st.stop()

    model = ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)

    message_content = []
    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext in ["jpg", "jpeg", "png"]:
            b64_img = encode_image(uploaded_file)
            message_content = [
                {"type": "text", "text": f"{prompt}\n(已上傳圖片：{uploaded_file.name})"},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64_img}"}
            ]
        elif file_ext == "pdf":
            pdf_text = extract_pdf_text(uploaded_file)
            message_content = f"{prompt}\n\n[檔案內容: {uploaded_file.name}]\n{pdf_text}"
        elif file_ext == "txt":
            txt_text = uploaded_file.getvalue().decode("utf-8")
            message_content = f"{prompt}\n\n[檔案內容: {uploaded_file.name}]\n{txt_text}"
    else:
        message_content = prompt

    st.session_state.messages.append(HumanMessage(content=message_content))
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file and file_ext in ["jpg", "jpeg", "png"]:
            st.image(uploaded_file)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 正在思考中...")
        
        try:
            response = model.invoke(st.session_state.messages)
            full_response = response.content
            message_placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
        except Exception as e:
            st.error(f"發生錯誤: {e}")