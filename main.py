import os
import sys

__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st

from datetime import datetime
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from io import BytesIO

# 👇 导入我们写好的核心模块 (此时 db_manager 里有 Chroma，parser 里有 RapidOCR，engine 里有 LLM)
from core.db_manager import DBManager
from core.llm_engine import LLMEngine
from core.document_parser import extract_text, chunk_text

# ================= 配置区 =================
API_KEY = st.secrets["DEEPSEEK_API_KEY"] 

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ================= 1. 初始化核心引擎 (加入 Session State 缓存防重载) =================
@st.cache_resource
def load_engines():
    """初始化数据库管家和 LLM 思考引擎，只运行一次"""
    db = DBManager()
    llm = LLMEngine(api_key=API_KEY)
    return db, llm


db_manager, llm_engine = load_engines()


# ================= 2. 辅助功能函数 =================
def text_to_audio_bytes(text):
    """将文本转为 MP3 字节流 (TTS)"""
    tts = gTTS(text=text, lang='zh-cn')
    fp = BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()


def generate_notes(messages):
    """生成精美的复习笔记 Markdown 文本"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note_content = f"📚 淮师大智能助手 - 专属复习笔记\n生成时间：{current_time}\n"
    note_content += "=" * 40 + "\n\n"

    # 过滤掉开头带有“欢迎语”的消息
    for msg in messages:
        if msg["role"] == "assistant":
            if "你好同学" in msg["content"] or "专属学霸导师" in msg["content"]:
                continue
            note_content += f"🤖 【导师解答】:\n{msg['content']}\n"
            note_content += "-" * 40 + "\n"
        elif msg["role"] == "user":
            note_content += f"👤 【我的问题】: {msg['content']}\n"

    return note_content


# ================= 3. Web 界面构建 =================
st.set_page_config(page_title="淮师大智能助手", page_icon="🏫", layout="wide")
st.title("🏫 校园全能智能助手 ")

tab_chat, tab_admin = st.tabs(["💬 聊天助手与视觉顾问", "🛠️ 长期知识库管理"])

# ================= 模块 A：聊天助手界面 (用于查阅本地文档与语音交互) =================
with tab_chat:
    with st.sidebar:
        st.header("⚙️ 模式与控制中心")
        current_mode = st.radio("🧠 选择助手大脑：", ["生活助手", "专业课导师"])

        enable_socratic = False
        if current_mode == "专业课导师":
            enable_socratic = st.toggle("💡 启发式教学 (不直接给答案)", value=False)

        st.divider()
        st.markdown("### 🎙️ 语音输入")
        # 语音转文字组件
        voice_input = speech_to_text(language='zh-CN', use_container_width=True, just_once=True, key='STT')
        enable_tts = st.toggle("🔊 开启语音播报", value=True)

        st.divider()
        st.markdown("### 💾 记忆管理")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🗑️ 清空当前对话", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        # 🌟 导出笔记按钮 (只有当对话有内容时才显示)
        if "messages" in st.session_state and len(st.session_state.messages) > 1:
            notes_text = generate_notes(st.session_state.messages)
            st.download_button(
                label="📥 导出复习笔记 (.txt)",
                data=notes_text,
                file_name=f"复习笔记_{datetime.now().strftime('%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    # 初始化聊天记录
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        welcome_msg = "你好同学！生活上遇到什么问题可以直接问我哦！" if current_mode == "生活助手" else "你好！我是你的专属学霸导师，复习到哪一章卡壳了？随时问我！"
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]

    # 显示历史对话
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入区域 (文字或语音)
    text_input = st.chat_input(f"当前模式：{current_mode}...")
    final_input = text_input or voice_input

    if final_input:
        with st.chat_message("user"):
            st.markdown(final_input)

        # 机器人思考区域
        with st.chat_message("assistant"):
            with st.spinner("正在查阅资料与思考中..."):
                # 💡 极其清晰的逻辑流：先查库(RAG)，再发给模型推理
                retrieved_context = db_manager.search(final_input, current_mode)
                reply = llm_engine.generate_reply(final_input, st.session_state.messages, retrieved_context,
                                                  current_mode, enable_socratic)

                # 显示模型回复
                st.markdown(reply)

                # 如果检索到了资料，显示折叠面板用于验证
                if retrieved_context:
                    with st.expander("👁️ 查看底层检索资料 (防自编验证)"):
                        st.info(retrieved_context)

                # 如果开启语音播报，生成音频
                if enable_tts:
                    with st.spinner("正在生成语音播报..."):
                        audio_bytes = text_to_audio_bytes(reply)
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        # 保存对话记录到 session state
        st.session_state.messages.append({"role": "user", "content": final_input})
        st.session_state.messages.append({"role": "assistant", "content": reply})

# ================= 模块 B：后台管理面板 (用于长期记忆构建) =================
with tab_admin:
    st.header("📚 长期知识库中央控制台 (在此可以管理生活与学习资料)")
    st.markdown("此处数据将永久保存至向量数据库 ChromaDB 中，供聊天助手调用。普通图片分析无需存入此处。")

    # 可视化大屏与清空
    st.markdown("### 📊 数据库长期记忆状态")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.info(f"**🏫 生活助手长期大脑**\n\n数据量：`{db_manager.count('生活助手')}` 条切片")
        if st.button("🔥 清空生活长期数据库", type="secondary", key="clear_life"):
            db_manager.clear("生活助手")
            st.toast("生活数据库已清空！", icon="✅")
            st.rerun()

    with col_d2:
        st.success(f"**🎓 专业课导师长期大脑**\n\n数据量：`{db_manager.count('专业课导师')}` 条切片")
        if st.button("🔥 清空学习长期数据库", type="secondary", key="clear_study"):
            db_manager.clear("专业课导师")
            st.toast("学习数据库已清空！", icon="✅")
            st.rerun()

    st.divider()

    # 上传新知识区域
    st.markdown("### 📤 上传新长期知识文档")
    target_brain = st.radio("请选择存入的数据库：", ["生活助手", "专业课导师"], horizontal=True)
    uploaded_file = st.file_uploader("拖拽或点击上传文档 (.txt, .docx, 纯文本PDF)", type=['txt', 'docx', 'pdf'],
                                     key='doc_uploader')

    if uploaded_file is not None:
        if st.button("🚀 开始解析并存入长期数据库", type="primary", key="start_learn"):
            with st.spinner("正在读取文件并榨取纯文本..."):
                extracted_text = extract_text(uploaded_file)
                if extracted_text and not extracted_text.startswith("Error"):
                    with st.spinner("正在进行语义切片并存入数据库..."):
                        chunks = chunk_text(extracted_text)
                        chunk_count = db_manager.ingest(chunks, target_brain, uploaded_file.name)
                    st.balloons()
                    st.success(f"🎉 学习完成！文件已存入【{target_brain}】的大脑中，新增 {chunk_count} 个记忆切片。")
                    st.rerun()
                else:
                    st.error(f"解析失败！Error: {extracted_text}")

    # ================= 🌟 🌟 终极整合：新增专门用于图片视觉内容的即时分析模块 =================
    # ================= 🌟 🌟 终极整合：智能视觉顾问 =================
    st.divider()
    st.markdown("### 👁️ 智能视觉顾问 (即时分析图片：课表或题目截图)")
    st.markdown("此处的图片内容抠字后，将**直接**交由 DeepSeek 大模型进行推理分析，**不存入**长期数据库。")

    # 选择分析意图
    image_analysis_type = st.radio("请选择图片分析意图：", ["解题思路引导 (不给最终答案)", "课程/时间安排规划"],
                                   horizontal=True)
    uploaded_image = st.file_uploader("上传截图或照片 (支持 .png, .jpg, .jpeg)", type=['png', 'jpg', 'jpeg'],
                                      key='img_uploader')

    if uploaded_image is not None:
        st.image(uploaded_image, caption="待分析图片", use_container_width=True)

        # 1. 独立的第一级按钮：只负责调用大模型并把结果存入临时记忆
        if st.button("🚀 开始即时视觉分析", type="primary", key="start_visual"):
            with st.spinner("正在启动 RapidOCR 视觉引擎疯狂抠字..."):
                ocr_text = extract_text(uploaded_image)

            if ocr_text:
                with st.spinner("正在请求 DeepSeek 导师大脑进行智能规划与解题分析..."):
                    analysis_result = llm_engine.generate_analysis_reply(ocr_text, image_analysis_type)
                    # 💡 核心修复：把结果存入 session_state 临时缓存，而不是直接显示
                    st.session_state.current_analysis = analysis_result
                    st.session_state.current_img_name = uploaded_image.name
            else:
                st.error("❌ 无法从图片中抠出有效文字，请确保图片清晰。")

        # 2. 独立的显示与第二级按钮逻辑：只要缓存里有结果，就一直显示！
        if st.session_state.get("current_analysis"):
            st.success("✅ 图片分析与规划完成！分析结果如下：")
            st.text_area("智能顾问分析报告：", value=st.session_state.current_analysis, height=400)

            # 这个按钮现在和上面的分析按钮是平级的了，点击它不会导致外层条件失效
            if st.button("➕ 将分析报告添加到聊天助手对话历史", type="secondary"):
                # 记录进对话历史
                st.session_state.messages.append(
                    {"role": "user",
                     "content": f"[视觉分析图片上传]: {st.session_state.current_img_name} ({image_analysis_type})"}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": st.session_state.current_analysis}
                )
                # 添加完毕后，清空临时缓存，让界面恢复初始状态
                st.session_state.current_analysis = None

                st.toast("🎉 已成功添加到聊天界面！请切换到【💬 聊天助手】查看。", icon="✅")
                st.rerun()  # 强制刷新页面，关闭当前的文本框
        else:
            st.error("❌ 无法从图片中抠出有效文字，请确保图片清晰且包含中文文本。")
