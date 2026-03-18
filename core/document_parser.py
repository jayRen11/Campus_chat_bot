import pdfplumber
import docx
import re
from pdf2image import convert_from_bytes
from PIL import Image
# 👇 换成坚如磐石的 Tesseract
import pytesseract

def extract_text(uploaded_file):
    """提取文件纯文本，自带智能视觉 OCR (稳定云端版)"""
    file_name = uploaded_file.name.lower()
    text = ""
    try:
        # --- 情况 A: 纯文本 TXT ---
        if file_name.endswith('.txt'):
            text = uploaded_file.read().decode('utf-8')
            
        # --- 情况 B: Word 文档 ---
        elif file_name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            
        # --- 情况 C: PDF 文件 ---
        elif file_name.endswith('.pdf'):
            file_bytes = uploaded_file.read()
            uploaded_file.seek(0) 
            
            with pdfplumber.open(uploaded_file) as pdf:
                total_pages = len(pdf.pages)
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted: 
                        text += extracted + "\n"
                        
            # 第二关：智能启动 OCR
            if len(text.strip()) < 50:
                print(f"\n🔍 检测到图片扫描件 PDF ({total_pages}页)，启动 Tesseract...")
                text = "" 
                
                for page_num in range(1, total_pages + 1):
                    images = convert_from_bytes(
                        file_bytes, 
                        first_page=page_num, 
                        last_page=page_num
                    )
                    if images:
                        # 核心修改：极其清爽的单行提取代码
                        page_text = pytesseract.image_to_string(images[0], lang='chi_sim')
                        text += page_text + "\n"
                                
        # --- 情况 D: 普通图片文件 ---
        elif file_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\n🔍 检测到图片文件上传，启动 Tesseract 视觉分析...")
            img_pil = Image.open(uploaded_file).convert('RGB')
            text = pytesseract.image_to_string(img_pil, lang='chi_sim')
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
