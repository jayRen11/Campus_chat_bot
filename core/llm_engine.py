from openai import OpenAI


class LLMEngine:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def generate_reply(self, user_input, chat_history, context, mode, socratic_mode=False):
        """通用 RAG 问答 (用于查阅本地文档)"""
        if mode == "生活助手":
            sys_role = "你是一个淮南师范学院的校园生活智能助手。请【严格且仅根据】参考信息回答。找不到就说不知道。"
        else:
            # 念诵严厉的 RAG 紧箍咒
            sys_role = "你是学霸导师。请【最高优先级且严格】依据我给你的【参考信息】来解答问题！参考信息中有答案不许瞎编；参考信息中没有提及才能用你的通用知识补充，但必须开头声明。"
            if socratic_mode:
                sys_role += "\n【启发式教学模式已开启】：不要直接给答案代码，先给提示反问引导思考。"

        system_prompt = f"{sys_role}\n\n【参考信息】：\n{context}"

        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.3 if socratic_mode else 0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"网络请求异常: {e}"

    # 👇 🌟 新增：专门用于图片视觉内容的分析逻辑
    def generate_analysis_reply(self, ocr_text, analysis_type):
        """针对图片抠出来的杂乱文字进行专门的逻辑分析与规划"""

        if not ocr_text:
            return "❌ 图片内容抠取失败，没有文字无法分析。"

        # 针对不同类型的图片定制不同的 Prompt 咒语
        if analysis_type == "解题思路":
            sys_prompt = (
                "你是金牌课外辅导名师。请严格【不要】给出最终答案、完整代码或详细数值结果。\n"
                "你的任务是针对参考信息里从图片中抠出来的题目文字进行分析。\n"
                "1. 分析题目考察的核心概念与知识点。\n"
                "2. 使用启发式、苏格拉底式的提问方式，一步步引导我，只给出下一步思考的‘思路’或‘提示词’。\n"
                "3. 不要超过 3 步规划，每次回答只推进一小步。"
            )
        else:  # analysis_type == "课程规划"
            sys_prompt = (
                "你是超级教务主任，拥有极强的逻辑分析能力。\n"
                "请将参考信息里从课表图片中抠出来的杂乱、无序的时间、课程、地点文字进行重组和结构化。\n"
                "1. 整理出一份清晰的、排版整洁的周一到周日课程时间大表。\n"
                "2. 标记出时间段比较空闲的‘黄金学习时段’。\n"
                "3. 基于此课表，给我制定一份精简的、包含复习和预习的高效周复习规划建议。"
            )

        # 构造最终的 Prompt 发给大模型
        user_prompt = f"这是从图片中通过 OCR 技术抠出来的文本，可能有些乱序，请基于这些信息：\n\n【参考信息】:\n{ocr_text}\n\n进行分析。"

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.5,  # 规划和思路需要稍微多一点思维发散
                max_tokens=1536  # 规划类输出字数可能较多
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"网络分析失败: {e}"