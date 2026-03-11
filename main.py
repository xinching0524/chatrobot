import streamlit as st
import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pypdf import PdfReader
from PIL import Image
import io

# 加載環境變數
load_dotenv()
import os
print(f"DEBUG - 抓到的 Key: {os.getenv('GOOGLE_API_KEY')}")
# 網頁配置
st.set_page_config(page_title="Gemini 繁體中文 AI 助手", page_icon="🤖", layout="wide")

# 自定義 CSS 讓介面更美觀
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def encode_image(image_file):
    """處理 Streamlit 上傳的圖片並轉換為 base64"""
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def extract_pdf_text(pdf_file):
    """從 Streamlit 上傳的 PDF 提取文字"""
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

# 側邊欄：配置與檔案上傳
with st.sidebar:
    st.title("⚙️ 設定與檔案")
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        api_key = st.text_input("輸入 Google API Key:", type="password")
    
    selected_model = st.selectbox("選擇模型:", ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-1.5-pro"])
    
    st.divider()
    st.subheader("📁 上傳檔案分析")
    uploaded_file = st.file_uploader("支援圖片 (JPG/PNG), PDF 或 TXT", type=["jpg", "jpeg", "png", "pdf", "txt"])
    
    if st.button("清除對話紀錄"):
        st.session_state.messages = [
            SystemMessage(content="你是 Gemini，由 Google 開發。請始終使用『繁體中文(台灣)』回答，語氣友善專業。")
        ]
        st.rerun()

st.title("🤖 Gemini 繁體中文 AI 助手")
st.caption("基於 LangChain 與 Google Gemini API 打造的網頁版對話機器人")

# 顯示聊天歷史 (排除 SystemMessage)
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            if isinstance(msg.content, list):
                # 處理含圖片的多模態輸入顯示
                for item in msg.content:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"])
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

    # 初始化模型
    model = ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)

    # 處理輸入內容
    message_content = []
    
    # 如果有上傳檔案，先處理檔案
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

    # 更新 UI 並將訊息加入歷史紀錄
    st.session_state.messages.append(HumanMessage(content=message_content))
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file and file_ext in ["jpg", "jpeg", "png"]:
            st.image(uploaded_file)

    # 獲取 AI 回應
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
            if "RESOURCE_EXHAUSTED" in str(e):
                st.warning("API 配額已達上限，請稍後再試。")
