# src/fixer.py
import re
import csv
import os
import time
from openai import OpenAI
from pylatexenc.latexwalker import LatexWalker
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.config_loader import load_config

# 简单配置一下log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

class FormulaFixer:
    """LaTeX 公式修复器：基于规则 + LLM 的混合修复引擎"""

    def __init__(self, config):
        """config应当传入config.get("fix")"""
        self.enable = config.get("enable", True)
        self.input_path = config.get("input_path")
        self.output_path = config.get("output_path")
        self.report_path = config.get("output_report_path")
        self.few_shot_path = config.get("few_shot_path")
        self.context_lines = config.get("context_lines", 5)
        self.gap_threshold = config.get("gap_threshold", 5)
        self.max_retries = config.get("max_retries", 3)
        self.max_workers = config.get("max_workers",5)

        ds_cfg = config.get("DeepSeek", {})
        self.api_url = ds_cfg.get("url", "https://api.deepseek.com/v1")
        self.api_key = ds_cfg.get("DEEPSEEK_API_KEY", "")
        # 加载 few-shot 样例
        self.base_messages = self._load_few_shots()

    def _load_few_shots(self):
        """加载 few-shot YAML，构造 system + few-shot 消息列表"""
        if not self.few_shot_path or not os.path.exists(self.few_shot_path):
            # 如果没有 few-shot，只返回 system prompt
            system_prompt = self._get_system_prompt()
            return [{"role": "system", "content": system_prompt}]

        with open(self.few_shot_path, 'r', encoding='utf-8') as f:
            few_shots = __import__('yaml').safe_load(f)   # 动态导入避免顶部依赖

        system_prompt = self._get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        for shot in few_shots:
            messages.append({"role": "user", "content": shot["user"]})
            messages.append({"role": "assistant", "content": shot["assistant"]})
        return messages

    @staticmethod
    def _get_system_prompt():
        return r"""你是一个顶级的 LaTeX 数学文档修复专家，专门负责将 MinerU 从高等代数教材 PDF 解析出的 Markdown 文本修复成适合 RAG 使用的干净格式。
用户会提供一段 MinerU 解析后的 Markdown 文本，其中需要修复的部分会被精确包裹在 [TARGET] 和 [/TARGET] 标签内。

你的任务：
- 只修复 [TARGET] ... [/TARGET] 之间的内容
- 修复识别错误，使 LaTeX 语义正确、结构清晰
- 最大程度保留原文数学含义
- 不在标签内的任何文字、公式、上下文都**必须原样保留**，即使你认为它有错也不要修改
- 严格按照以下规则修复：
  1. 长除法、综合除法、竖式、表格竖式 → 全部替换为[省略竖式计算过程]
  2. 下标错误（a01、a04 等）→ 修复为 a_0^1、a_0^4
  3. 缺失花括号、\\overline 识别错误、\\textcircled 识别错误
  4. 其他最小必要语法清理（不要改变语义）
- 如果 [TARGET] 内的内容已经完美，无需任何修改，则 FIX_RESULT 直接返回和 FIX_ORIGIN 完全相同的内容

输出要求（必须严格遵守）：
- 不要输出 JSON，不要输出任何额外的解释文字，不要加 Markdown 代码块。
- 返回的文本被两个标签所包裹：[FIX_ORIGIN]和[/FIX_ORIGIN]用来包裹原始文本，[FIX_RESULT]和[/FIX_RESULT]用来包裹你修复后的文本。
- [FIX_ORIGIN] 之间的内容必须与用户提供的 [TARGET] 内容完全一致，包括所有空白、换行和定界符，不做任何修改。

示例输出：
[FIX_ORIGIN]
原始文本
[/FIX_ORIGIN]

[FIX_RESULT]
修改后文本
[/FIX_RESULT]
"""


    @staticmethod
    def _detect_issues(block):
        """多层规则判断是否需要修复
        返回 True 表示该块需要修复"""
        content = block.strip()

        # --- 优先级 1：明确的 OCR 幻觉标识 ---
        # 1. 矩阵/行列式内部不该出现的 hline
        if re.search(r'\\left\s*\|[\s\S]*?\\hline', content) or \
           re.search(r'\\begin\{array\}[\s\S]*?\\hline', content):
            return True
        # 数组当中不应该出现的|
        if re.search(r'\\begin\{array\}[\s\S]*?\\left\|', content):
            return True
        # 2. 连续下划线错误
        if re.search(r'\_\{?\s*\_', content):
            return True
        # 3. 典型的初等变换文字识别错误
        if re.search(r'\\text\s*\{(行|列|次|个)\}', content):
            return True
        if re.search(r'\\text\s*\{\s*\}', content):
            return True
        # 4. 下标连写错误 (a01 -> a_0^1)
        if re.search(r'[a-zA-Z]0[1-9]', content):
            return True
        # 5. 特殊符号检测
        if re.search(r'\\textcircled', content) or re.search(r'[①-⑨]', content):
            return True
        # 5.2 \overline 仅在矩阵环境中触发
        matrix_env_pattern = r'\\begin\{(?:array|pmatrix|bmatrix|vmatrix|Vmatrix|matrix)\}'
        if re.search(matrix_env_pattern, content) and re.search(r'\\overline', content):
            return True

        # --- 优先级 2：结构性崩溃 ---
        # 花括号不匹配
        open_braces = len(re.findall(r'(?<!\\)\{', content))
        close_braces = len(re.findall(r'(?<!\\)\}', content))
        if open_braces != close_braces:
            return True
        # \left 与 \right 配对
        if content.count(r'\left') != content.count(r'\right'):
            return True

        # 语法树检查（兜底）
        try:
            latex_content = content
            if content.startswith('$$') and content.endswith('$$'):
                latex_content = content[2:-2]
            elif content.startswith('$') and content.endswith('$'):
                latex_content = content[1:-1]
            LatexWalker(latex_content).get_latex_nodes()
        except Exception:
            return True

        # 表格直接扔进去
        if '<table' in content.lower():
            return True

        return False

    @staticmethod
    def _expand_boundary(content, start, end, gap_threshold=10):
        """是否存在临近的数学块，若存在，则合并一同修复
        content:当前文本
        start, end:当前块的起止位置
        gap_threshold:距离start或end多少字符内，是否存在数学符号，若存在，则合并
        """
        block_pattern = re.compile(
            r'(\$\$.*?\$\$|\\\[.*?\\\]|\$.*?\$|<table.*?<\/table>)',
            re.DOTALL
        )
        blocks = [(m.start(), m.end()) for m in block_pattern.finditer(content)]
        if not blocks:
            line_start = content.rfind('\n', 0, start) + 1
            line_end = content.find('\n', end)
            if line_end == -1:
                line_end = len(content)
            return line_start, line_end

        # 找到包含原始区域或最近的块
        orig_idx = None
        for i, (b_start, b_end) in enumerate(blocks):
            if b_start <= start < b_end or b_start <= end < b_end:
                orig_idx = i
                break
        if orig_idx is None:
            for i, (b_start, _) in enumerate(blocks):
                if b_start > start:
                    orig_idx = i
                    break
            if orig_idx is None:
                orig_idx = len(blocks) - 1

        # 向左合并
        merged_start_idx = orig_idx
        for i in range(orig_idx - 1, -1, -1):
            gap = blocks[merged_start_idx][0] - blocks[i][1]
            if gap < gap_threshold:
                merged_start_idx = i
            else:
                break
        # 向右合并
        merged_end_idx = orig_idx
        for i in range(orig_idx + 1, len(blocks)):
            gap = blocks[i][0] - blocks[merged_end_idx][1]
            if gap < gap_threshold:
                merged_end_idx = i
            else:
                break

        new_start = blocks[merged_start_idx][0]
        new_end = blocks[merged_end_idx][1]
        # 对齐到完整行
        line_start = content.rfind('\n', 0, new_start) + 1
        line_end = content.find('\n', new_end)
        if line_end == -1:
            line_end = len(content)
        return line_start, line_end

    def _call_llm_repair(self, context_content):
        """
        调用 DeepSeek API 修复单个片段。
        context_content 是包含 [TARGET]...[/TARGET] 的完整上下文。
        返回字典 {"origin": str, "fix": str} 或 None。
        """
        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        for attempt in range(self.max_retries):
            try:
                messages = self.base_messages + [{"role": "user", "content": context_content}]
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.1,
                    timeout=45
                )
                res_text = response.choices[0].message.content

                origin_match = re.search(r'\[FIX_ORIGIN\]\s*(.*?)\s*\[/FIX_ORIGIN\]', res_text, re.DOTALL)
                fix_match = re.search(r'\[FIX_RESULT\]\s*(.*?)\s*\[/FIX_RESULT\]', res_text, re.DOTALL)

                if origin_match and fix_match:
                    return {
                        "origin": origin_match.group(1).strip(),
                        "fix": fix_match.group(1).strip()
                    }
            except Exception as e:
                time.sleep((attempt + 1) * 2)
        return None

    def _fix(self, text):
        if not self.enable:
            return text, []

        # 1. 移除图片行
        text = re.sub(r'^!\[.*\]\(.*\).*\n?', '', text, flags=re.MULTILINE)

        # 2. 找出所有核心块
        core_regex = r'(\$\$.*?\$\$|\\\[.*?\\\]|<table>.*?</tr>)'
        matches = list(re.finditer(core_regex, text, flags=re.DOTALL))
        logging.info(f"发现 {len(matches)} 个核心块。")

        # 3. 收集需要修复的区域（扩展后）
        regions = []
        for match in matches:
            if self._detect_issues(match.group()):
                start, end = self._expand_boundary(text, match.start(), match.end(), self.gap_threshold)
                regions.append((start, end))

        if not regions:
            logging.info("没有需要修复的区域。")
            return text, []

        # 4. 合并重叠或相邻区域
        regions.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = regions[0]
        for s, e in regions[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        logging.info(f"合并后共 {len(merged)} 个修复区域。")

        # 准备任务列表
        tasks = []
        for start, end in merged:
            target = text[start:end]
            prefix = text[:start].split('\n')[-self.context_lines:]
            suffix = text[end:].split('\n')[:self.context_lines]
            prompt = f"{''.join(prefix)}\n\n[TARGET]\n{target}\n[/TARGET]\n\n{''.join(suffix)}"
            # 记录原始位置，方便后续回填
            tasks.append({
                "start": start,
                "end": end,
                "target": target,
                "prompt": prompt
            })

        results_map = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(self._call_llm_repair, t["prompt"]): t for t in tasks
            }
            
            # 使用 tqdm 显示进度
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="并行修复中"):
                task = future_to_task[future]
                try:
                    res = future.result()
                    # 以原始 start 位置为 Key 存入字典
                    results_map[task["start"]] = {
                        "fixed": res["fix"] if res else None,
                        "res_obj": res,
                        "task_info": task
                    }
                except Exception as e:
                    logging.error(f"线程执行异常: {e}")
                    results_map[task["start"]] = {"fixed": None, "res_obj": None, "task_info": task}

        # 6. 逆序回填
        new_content = text
        report_items = []
        sorted_starts = sorted(results_map.keys(), reverse=True)

        for s in sorted_starts:
            item = results_map[s]
            t = item["task_info"]
            fixed = item["fixed"]
            
            if fixed:
                new_content = new_content[:t["start"]] + fixed + new_content[t["end"]:]
                status = "success"
            else:
                status = "failed"
            
            report_items.append({
                "original": t["target"],
                "fixed": fixed or "",
                "status": status
            })

        return new_content, report_items
    
    def _process_file(self):
        """内置的文件处理逻辑"""
        if not self.input_path or not os.path.exists(self.input_path):
            logging.error(f"输入文件不存在: {self.input_path}")
            return False

        logging.info(f"开始处理文件: {self.input_path}")
        with open(self.input_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        fixed_text, report = self._fix(raw_content)

        # 写入结果
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(fixed_text)
            logging.info(f"修复完成，保存至: {self.output_path}")

        # 写入报告
        if self.report_path and report:
            self._save_report(report)
        
        return True

    def _save_report(self, report):
        """内部报告保存逻辑"""
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=report[0].keys())
            writer.writeheader()
            writer.writerows(report)
        logging.info(f"报告已生成: {self.report_path}")

    def run(self):
        """对外暴露的唯一启动方法"""
        if not self.enable:
            logging.warning("FormulaFixer 已禁用。")
            return
        
        try:
            self.output_path
            if os.path.exists(self.output_path):
                logging.info(f"从存在修复后文件：{self.output_path}")
                return
            else:
                start_time = time.time()
                success = self._process_file()
                if success:
                    logging.info(f"任务耗时: {time.time() - start_time:.2f}s")
        except Exception as e:
            logging.error(f"处理过程中发生异常: {str(e)}")

def main():
    # 1. 加载配置
    config = load_config()
    
    # 2. 初始化
    fixer = FormulaFixer(config["fix"])
    
    # 3. 执行
    fixer.run()


if __name__ == "__main__":
    main()