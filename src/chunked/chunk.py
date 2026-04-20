import re
import json
import os
from transformers import AutoTokenizer
import logging

from src.utils.config_loader import load_config

# 简单配置一下log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

class ChunkProcessor:
    def __init__(self, chunk_config):
        """
        初始化配置与资源
        chunk_config: 传入 global_config.get("chunked")
        """
        # 路径配置
        self._input_path = chunk_config.get("input_path")
        self._output_path = chunk_config.get("output_path")
        
        # 模型与阈值配置
        embed_cfg = chunk_config.get("embedding_model", {})
        self._tokenizer_path = embed_cfg.get("model_path")
        self._long_threshold = embed_cfg.get("LONG_THRESHOLD", 800)
        self._short_threshold = embed_cfg.get("SHORT_THRESHOLD", 50)
        
        # 初始化组件
        self._tokenizer = self._init_tokenizer()
        self._init_regex()
        
        # 规范化映射
        self._TYPE_NORM_MAP = {
            "Hamilton-Cayley定理": "定理",
            "命题": "定理", 
            "证法": "证明",
            "解法": "解"
        }

    def _init_tokenizer(self):
        """初始化分词器"""
        try:
            return AutoTokenizer.from_pretrained(self._tokenizer_path)
        except Exception as e:
            logging.info(f"警告：未找到分词器({e})，将使用字符长度粗略估算。")
            return None

    def _init_regex(self):
        """预编译所有正则表达式"""
        # 路径触发正则
        self._re_chapter = re.compile(r"^#+\s*第([0-9一二三四五六七八九十]+)章\s*(.*)")
        self._re_section = re.compile(r"^#*\s*(\d+\.\d+)\s+(.*)")
        self._re_subsection = re.compile(r"^#*\s*(\d+\.\d+\.\d+)\s+(.*)")
        self._re_topic = re.compile(r"^#*\s*([一二三四五六七八九十]、)\s*(.*)")
        self._re_special = re.compile(r"^#*\s*(习题|补充题)(.*)")
        self._re_app_world = re.compile(r"^#*\s*(应用小天地)(.*)")
        
        # 习题与关键字正则
        self._re_question = re.compile(r"^\d+[\.、]\s*")
        self._re_keywords = re.compile(
            r"^(?:[\*\-\+]\s*)?【?("
            r"Hamilton-Cayley定理|定理|定义|命题|推论|引理|例|结论|性质|"
            r"证明|证法[0-9一二三四五六七八九十]|解|解法[0-9一二三四五六七八九十]"
            r")(?:\s*\d+(?:\.\d+)*)?"
            r"】?\s*[:：]?", re.MULTILINE
        )
        
        # 切分逻辑正则
        self._re_logic_split = r'(\n\s*(?:由于|因此|从而|综上所述|由上式得|同理|于是|充分性|必要性|注意|容易验证|我们来证|假设|剩下只要证明|点评|直接验证|由此得出|情形[0-9一二三四五六七八九十]|注|任取))'
        # 这是最后一步，是在切分不动，按照中文符号切分，此时后续会增加overlap
        self._re_punct_split = r'([。；\n])'

    def _calc_tokens(self, path_str, content):
        """计算 Token 数量"""
        text_to_encode = f"[{path_str}]\n{content}" if path_str else content
        if self._tokenizer:
            return len(self._tokenizer.encode(text_to_encode, add_special_tokens=True))
        return len(text_to_encode) // 2

    def _get_math_heavy(self, text):
        """数学公式密度判断"""
        return text.count('$') > 3

    def _safe_split_content(self, text):
        """带数学公式保护的逻辑切分引擎"""
        math_blocks = []
        def _replacer(match):
            math_blocks.append(match.group(0))
            return f" __MATH_BLOCK_{len(math_blocks)-1}__ "

        # 1. 保护 LaTeX
        masked = re.sub(r'\$\$.*?\$\$', _replacer, text, flags=re.DOTALL)
        masked = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', _replacer, masked, flags=re.DOTALL)

        # 2. 逻辑词切分
        raw_logic_pieces = re.split(self._re_logic_split, masked)
        logic_atomics = []
        current_text = raw_logic_pieces[0]
        if current_text.strip():
            logic_atomics.append((current_text, False))
            
        for i in range(1, len(raw_logic_pieces), 2):
            combined = raw_logic_pieces[i] + raw_logic_pieces[i+1]
            if combined.strip():
                logic_atomics.append((combined, True))

        # 3. 长度细分
        final_atomics = []
        for piece_text, is_logic in logic_atomics:
            if len(piece_text) > 300:
                sub_pieces = re.split(self._re_punct_split, piece_text)
                temp = ""
                is_first = True
                for p in sub_pieces:
                    if re.match(self._re_punct_split, p):
                        temp += p
                        final_atomics.append((temp, is_logic if is_first else False))
                        temp = ""; is_first = False
                    else:
                        temp += p
                if temp.strip():
                    final_atomics.append((temp, is_logic if is_first else False))
            else:
                final_atomics.append((piece_text, is_logic))

        # 4. 还原公式
        restored = []
        for atom_text, is_logic in final_atomics:
            res = atom_text
            for j, mb in enumerate(math_blocks):
                res = res.replace(f" __MATH_BLOCK_{j}__ ", mb)
            if res.strip():
                restored.append((res.strip(), is_logic))
        return restored

    def _extract_tail_overlap(self, text, max_chars=200):
        """提取重叠部分上下文"""
        last_punct = max(text.rfind('。'), text.rfind('；'))
        if last_punct != -1:
            overlap = text[last_punct+1:].lstrip()
            return overlap[-max_chars:] if len(overlap) > max_chars else overlap
        return ""

    def _parse_markdown(self):
        """阶段一：解析 Markdown 并提取原始块"""
        raw_chunks = []
        buffer = []
        state = {
            "path": {"ch": "", "sec": "", "subsec": "", "topic": ""},
            "ch_num": "",
            "type": "text",
            "is_in_ex": False,
            "ex_count": 0
        }

        def _save_buffer():
            content = "\n".join(buffer).strip()
            if not content: return
            p = state["path"]
            path_str = " > ".join([v for v in [p["ch"], p["sec"], p["subsec"], p["topic"]] if v])
            raw_chunks.append({
                "metadata": {
                    "path": path_str, "type": state["type"], 
                    "is_logic_block": state["type"] in ["证明", "解"],
                    "chapter": state["ch_num"], "is_math_heavy": self._get_math_heavy(content)
                },
                "raw_content": content 
            })

        with open(self._input_path, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln: continue

                # 匹配检测
                m_ch = self._re_chapter.match(ln); m_sec = self._re_section.match(ln)
                m_sub = self._re_subsection.match(ln); m_top = self._re_topic.match(ln)
                m_spc = self._re_special.match(ln); m_app = self._re_app_world.match(ln)
                m_key = self._re_keywords.match(ln)
                
                is_path = any([m_ch, m_sec, m_sub, m_top, m_spc, m_app])
                
                # 维护习题区状态
                if m_spc: state["is_in_ex"] = True; state["ex_count"] = 0
                elif m_ch or m_sec or m_app: state["is_in_ex"] = False  # 遇到新的章、节、或应用小天地，退出习题区

                # 对于“习题”中的“证明|解”，忽略其类型触发，视为普通文本
                norm_type = None
                if m_key:
                    norm_type = m_key.group(1)
                    for k, v in self._TYPE_NORM_MAP.items():
                        if k in norm_type: norm_type = v; break
                    if state["is_in_ex"] and norm_type in ["证明", "解"]:
                        m_key = None; norm_type = None

                # 判断是否在is_in_ex，也就是判断是否为“习题”，每一题都切分一次
                is_q_trig = False
                if state["is_in_ex"] and not m_key and not is_path:
                    if self._re_question.match(ln):
                        state["ex_count"] += 1
                        if (state["ex_count"] - 1) % 1 == 0: is_q_trig = True

                # 触发保存
                if is_path or m_key or is_q_trig:
                    if buffer: _save_buffer(); buffer = []

                # 更新 Metadata
                if m_ch:
                    state["ch_num"] = m_ch.group(1)
                    state["path"].update({"ch": f"第{state['ch_num']}章 {m_ch.group(2)}", "sec": "", "subsec": "", "topic": ""})
                    state["type"] = "chapter_header"
                elif m_sec:
                    state["path"].update({"sec": f"{m_sec.group(1)} {m_sec.group(2)}", "subsec": "", "topic": ""})
                    state["type"] = "section_header"
                elif m_app:
                    state["path"].update({"sec": f"{m_app.group(1)} {m_app.group(2)}", "subsec": "", "topic": ""})
                    state["type"] = "application_world"
                elif m_sub:
                    state["path"].update({"subsec": f"{m_sub.group(1)} {m_sub.group(2)}", "topic": ""})
                    state["type"] = "subsection_header"
                elif m_top:
                    state["path"]["topic"] = f"{m_top.group(1)} {m_top.group(2)}"
                    state["type"] = "topic_header"
                elif m_spc:
                    state["path"].update({"sec": f"{m_spc.group(1)} {m_spc.group(2)}", "subsec": "", "topic": ""})
                    state["type"] = "exercise"
                elif m_key: state["type"] = norm_type
                elif is_q_trig: state["type"] = "exercise"
                
                if not is_path: buffer.append(ln)

        if buffer: _save_buffer()
        return raw_chunks

    def _post_process(self, raw_chunks):
        """阶段二：合并与超长块熔断"""
        # 1. 证明合并逻辑
        i = 0
        while i < len(raw_chunks) - 1:
            if raw_chunks[i]["metadata"]["type"] == "证明" and raw_chunks[i+1]["metadata"]["type"] == "证明":
                if i > 0:
                    raw_chunks[i-1]["raw_content"] += "\n" + raw_chunks[i]["raw_content"]
                    raw_chunks.pop(i)
                else: i += 1
            else: i += 1

        # 2. 熔断拆分
        i = 0; group_counter = 0
        while i < len(raw_chunks):
            c_path = raw_chunks[i]["metadata"]["path"]
            c_content = raw_chunks[i]["raw_content"]

            if self._calc_tokens(c_path, c_content) > self._long_threshold:
                atomics = self._safe_split_content(c_content)
                if len(atomics) > 1:
                    new_subs = []; curr_buf = []
                    for atom_text, is_logic in atomics:
                        # 如果加入这一块超长了，或者这一块是“逻辑块”且 buffer 不为空（强制逻辑换块）
                        # 这里你可以选择：逻辑词是【触发】换块，还是仅仅【不带Overlap】
                        # 下面逻辑：仅当长度超限时才切分，但切分时检查 is_logic
                        test_content = "\n".join(curr_buf + [atom_text])
                        if self._calc_tokens(c_path, test_content) > self._long_threshold and curr_buf:
                            new_subs.append("\n".join(curr_buf).strip())
                            if is_logic:
                                # 如果下一段是逻辑起始，不需要 Overlap
                                curr_buf = [atom_text]
                            else:
                                # 如果下一段是普通标点切分，提取上一段末尾作为 Overlap
                                overlap = self._extract_tail_overlap(curr_buf[-1])
                                curr_buf = [overlap, atom_text] if overlap else [atom_text]
                        else: curr_buf.append(atom_text)
                    if curr_buf: new_subs.append("\n".join(curr_buf).strip())

                    # 替换原始的长块
                    if len(new_subs) > 1:
                        group_id = f"group_{group_counter}"; group_counter += 1
                        base_meta = raw_chunks.pop(i)["metadata"]
                        for j, sub_content in enumerate(new_subs):
                            nm = base_meta.copy()
                            nm.update({"block_id": group_id, "part_id": f"{j+1}/{len(new_subs)}", "part": f"{j+1}/{len(new_subs)}"})
                            raw_chunks.insert(i + j, {"metadata": nm, "raw_content": sub_content})
                        i += len(new_subs); continue
            i += 1
        return raw_chunks

    def _assemble(self, raw_chunks):
        """阶段三：最终组装与格式化"""
        final_chunks = []
        for idx, c in enumerate(raw_chunks):
            path_str = c["metadata"]["path"]
            content = c["raw_content"]
            prefix = f"[{path_str}]"
            if "part" in c["metadata"]:
                prefix = f"[{path_str} (续 {c['metadata']['part']})]"
            
            final_chunks.append({
                "id": str(idx),
                "metadata": c["metadata"],
                "text": f"{prefix}\n{content}" if path_str else content,
                "token_count": self._calc_tokens(path_str, content)
            })
        return final_chunks

    def _report(self, chunks):
        """分析报告"""
        longs = [c for c in chunks if c["token_count"] > self._long_threshold]
        shorts = [c for c in chunks if c["token_count"] < self._short_threshold]
        logging.info(f"处理完成.总块数: {len(chunks)},超长块: {len(longs)},超短块: {len(shorts)}")

    def run(self):
        """对外暴露的唯一调用接口"""
        if not os.path.exists(self._input_path):
            logging.error(f"错误: 输入文件不存在 {self._input_path}")
            return
        
        if not os.path.exists(self._output_path):
            logging.info(f"输出文件已存在 {self._input_path}")
            return
            
        raw_chunks = self._parse_markdown()
        processed_chunks = self._post_process(raw_chunks)
        final_chunks = self._assemble(processed_chunks)
        
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            
        self._report(final_chunks)
        return final_chunks

def main():
    config = load_config()
    chunk_processor = ChunkProcessor(config.get("chunked"))
    chunk_processor.run()

if __name__ == "__main__":
    main()