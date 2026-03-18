import pdfplumber
import docx
import re
from pdf2image import convert_from_bytes
import numpy as np
# 👇 新增两个库：处理图片流和图片格式转换
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# 1. 初始化 RapidOCR
ocr = RapidOCR()




def extract_text(uploaded_file):
    """提取文件纯文本，自带智能视觉 OCR (新增图片支持)"""
    file_name = uploaded_file.name.lower()  # 统一转小写判断后缀
    text = ""
    try:
        # --- 情况 A: 纯文本 TXT ---
        if file_name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')

        # --- 情况 B: Word 文档 ---
        elif file_name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])

        # --- 情况 C: PDF 文件 (含图片扫描件支持) ---
        elif file_name.endswith('.pdf'):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0)

            # 第一关：常规提取
            with pdfplumber.open(uploaded_file) as pdf:
                total_pages = len(pdf.pages)
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

            # 第二关：智能启动 OCR
            if len(text.strip()) < 50:
                print(f"\n🔍 检测到图片扫描件 PDF ({total_pages}页)，启动 RapidOCR...")
                text = ""

                for page_num in range(1, total_pages + 1):
                    images = convert_from_bytes(file_bytes, first_page=page_num, last_page=page_num)

                    if images:
                        img_np = np.array(images[0])
                        result, _ = ocr(img_np)
                        if result:
                            for line in result:
                                text += line[1] + "\n"

        # --- 情况 D (🌟新增): 普通图片文件 (课表/题目图片) ---
        elif file_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n🔍 检测到图片文件上传，正在启动 RapidOCR 视觉分析...")
            # 利用 PIL 库把用户上传的文件流转成 RapidOCR 需要的格式
            img_pil = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(img_pil)

            # 执行 OCR 抠字
            result, _ = ocr(img_np)
            if result:
                for line in result:
                    text += line[1] + "\n"
                print(f"✅ 图片文字抠取完成，共抠出 {len(text.strip())} 个字。")

        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    except Exception as e:
        return f"Error: 提取失败 - {e}"


def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks