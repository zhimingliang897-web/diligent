import ast
import json
import operator
import re
from datetime import datetime
from pathlib import Path

import httpx
from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """获取当前日期和时间，包括星期几。
    【强制要求】当用户询问任何日期相关问题时，必须先调用此工具获取当前真实日期。
    系统时间就是真实时间，不受模型知识截止日期影响。
    当用户问"今天几号"、"某天是否到来"、"现在是哪一年"等问题时，必须调用此工具。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday})")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式并返回结果。
    用于任何算术计算，支持 +, -, *, /, **（幂运算）和括号。
    示例输入: '2 + 3', '(10 + 5) * 2', '2 ** 10'"""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量: {node.value}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return allowed_operators[op_type](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return allowed_operators[op_type](_eval(node.operand))
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as e:
        return f"计算 '{expression}' 时出错: {e}"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。
    当查询时事新闻、实时数据（如今天A股指数、天气、比赛结果）时使用此工具。
    【关键提示】查询实时动态数据时，务必在查询词 query 中显式加上具体的年月日时间（例如：'2026年3月17日 A股上证指数'），这样能确保搜索引擎返回最新最准的结果。
    返回标题、摘要和来源链接。部分请求可能因网络原因耗时，支持中英文查询。"""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        if not hits:
            return f"未找到关于 '{query}' 的搜索结果，请尝试换个关键词。"
        lines = [
            f"{i}. 【{r.get('title', '')}】\n{r.get('body', '')}\n{r.get('href', '')}"
            for i, r in enumerate(hits, 1)
        ]
        return "\n\n".join(lines)
    except Exception as e:
        return f"搜索出错: {e}"


@tool
def remember_user_fact(key: str, value: str) -> str:
    """记住关于用户的长期事实或偏好信息以供未来跨对话使用。
    当你了解到关于用户的新事实时（例如：用户的名字，职业，爱好，习惯等），务必调用此工具。
    参数说明：
    - key: 事实名称，尽量精简，如 "名字", "职业", "编程语言偏好"
    - value: 事实内容，如 "阿亮", "全栈工程师", "Python"
    """
    from agent.memory.profile import update_user_fact
    return update_user_fact(key, value)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息。
    当用户询问天气、气温、是否下雨等天气相关问题时使用此工具。
    参数说明：
    - city: 城市名（支持中文或英文，如 "北京"、"Shanghai"、"New York"）
    """
    try:
        # wttr.in 免费天气 API，无需 Key
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        response = httpx.get(url, timeout=10.0, headers={"User-Agent": "curl/7.0"})
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        area_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        desc = current["lang_zh"][0]["value"] if current.get("lang_zh") else current["weatherDesc"][0]["value"]
        wind_speed = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        visibility = current["visibility"]

        return (
            f"📍 {area_name}, {country}\n"
            f"🌡️ 当前气温: {temp_c}°C（体感 {feels_like}°C）\n"
            f"🌤️ 天气状况: {desc}\n"
            f"💧 湿度: {humidity}%\n"
            f"💨 风速: {wind_speed} km/h，风向: {wind_dir}\n"
            f"👁️ 能见度: {visibility} km"
        )
    except httpx.HTTPError as e:
        return f"天气查询网络错误: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"天气数据解析失败（城市名可能不正确）: {e}"
    except Exception as e:
        return f"天气查询失败: {e}"


# ──────────────────────── 常驻免费扩展工具 ────────────────────────

@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算。支持常见长度、重量、温度、面积、体积单位。
    当用户询问单位换算时使用此工具。
    参数: value 数值, from_unit 源单位（如 m, km, kg, g, °C, °F）, to_unit 目标单位。"""
    from_unit = from_unit.strip().lower()
    to_unit = to_unit.strip().lower()
    # 长度 -> 米
    length_to_m = {"m": 1, "km": 1000, "dm": 0.1, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "mile": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254, "inch": 0.0254}
    # 重量 -> 千克
    mass_to_kg = {"kg": 1, "g": 0.001, "mg": 1e-6, "t": 1000, "ton": 1000, "lb": 0.453592, "oz": 0.0283495}
    try:
        if from_unit in length_to_m and to_unit in length_to_m:
            m = value * length_to_m[from_unit]
            result = m / length_to_m[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        if from_unit in mass_to_kg and to_unit in mass_to_kg:
            kg = value * mass_to_kg[from_unit]
            result = kg / mass_to_kg[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        if from_unit in ("°c", "c", "摄氏度") and to_unit in ("°f", "f", "华氏度"):
            f = value * 9 / 5 + 32
            return f"{value}°C = {f:.2f}°F"
        if from_unit in ("°f", "f", "华氏度") and to_unit in ("°c", "c", "摄氏度"):
            c = (value - 32) * 5 / 9
            return f"{value}°F = {c:.2f}°C"
        return f"暂不支持从 {from_unit} 到 {to_unit} 的换算，支持: 长度(m,km,cm,mm,ft,in), 重量(kg,g,lb), 温度(°C,°F)。"
    except Exception as e:
        return f"换算出错: {e}"


def _is_private_url(url: str) -> bool:
    """检测 URL 是否指向私有/内网地址（防 SSRF）。"""
    import ipaddress
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
        if host.lower() in ("localhost", ""):
            return True
        addr = ipaddress.ip_address(host)
        return addr.is_private or addr.is_loopback or addr.is_link_local
    except ValueError:
        return False


@tool
def fetch_webpage(url: str, max_chars: int = 8000) -> str:
    """抓取网页正文内容并返回纯文本（用于摘要或问答）。
    当用户提供链接并希望了解页面内容时使用此工具。
    参数: url 网页地址, max_chars 最多返回字符数（默认8000）。仅支持 http/https。"""
    if not url.strip().lower().startswith(("http://", "https://")):
        return "仅支持 http 或 https 链接。"
    if _is_private_url(url):
        return "安全限制: 不允许访问内网或本地地址。"
    try:
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", text, flags=re.I)
        text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text or "未能提取到正文内容。"
    except Exception as e:
        return f"抓取失败: {e}"


@tool
def format_json(json_string: str) -> str:
    """格式化或校验 JSON 字符串。若输入合法则返回美化后的 JSON；否则返回错误信息。"""
    try:
        obj = json.loads(json_string)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {e}"
    except Exception as e:
        return f"处理出错: {e}"


@tool
def summarize_text(text: str, max_sentences: int = 5) -> str:
    """对长文本做简短摘要：取前 N 句或前 500 字。
    当用户希望概括、总结一段文字时使用此工具。
    参数: text 原文, max_sentences 最多保留句数（默认5）。"""
    if not text or not text.strip():
        return "没有可摘要的文本。"
    text = text.strip()
    # 按句号、问号、感叹号分句
    sentences = re.split(r"[。！？!?]\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= max_sentences:
        return text[:500] + ("..." if len(text) > 500 else "")
    result = "".join(s + "。" for s in sentences[:max_sentences])
    return result[:500] + ("..." if len(result) > 500 else result)


@tool
def translate_text(text: str, target_lang: str = "中文") -> str:
    """将文本翻译成目标语言。依赖 LLM 进行一次调用，会消耗少量 token。
    当用户明确要求翻译时使用此工具。参数: text 待翻译文本, target_lang 目标语言（如 中文、英文）。"""
    try:
        from agent.llm import get_llm
        llm = get_llm()
        prompt = f"请将以下内容翻译成{target_lang}，只输出翻译结果，不要解释。\n\n{text}"
        msg = llm.invoke(prompt)
        return (msg.content or "").strip() or "翻译未返回内容。"
    except Exception as e:
        return f"翻译失败: {e}"


def _make_text_to_image():
    """文生图（千问/万相）。仅当 ENABLE_TEXT_TO_IMAGE 为真时在 get_all_tools 中加入。"""
    import logging as _logging
    _t2i_log = _logging.getLogger("agent.t2i")

    @tool
    def text_to_image(prompt: str, size: str = "1328*1328") -> str:
        """根据文字描述生成图片（文生图）。调用通义万相/千问图像 API，返回图片 URL。
        当用户要求"画一张图""根据描述生成图片"时使用。参数: prompt 描述内容, size 可选 1328*1328 / 1664*928 / 928*1664 / 1472*1104 / 1104*1472。"""
        import tempfile, os
        try:
            from agent.config import DASHSCOPE_API_KEY, TEXT_TO_IMAGE_MODEL
            size = (size or "1328*1328").replace("x", "*").replace("×", "*")
            payload = {
                "model": TEXT_TO_IMAGE_MODEL,
                "input": {
                    "messages": [{
                        "role": "user",
                        "content": [{"text": prompt}],
                    }],
                },
                "parameters": {
                    "size": size,
                    "n": 1,
                    "prompt_extend": True,
                    "watermark": False,
                },
            }
            _t2i_log.info("文生图请求: model=%s, size=%s", TEXT_TO_IMAGE_MODEL, size)
            resp = httpx.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers={
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120.0,
            )
            if resp.status_code != 200:
                _t2i_log.error("文生图 HTTP %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
            data = resp.json()
            image_url = data["output"]["choices"][0]["message"]["content"][0]["image"]

            img_resp = httpx.get(image_url, timeout=60.0)
            img_resp.raise_for_status()
            tmp_dir = os.path.join(tempfile.gettempdir(), "myagent_images")
            os.makedirs(tmp_dir, exist_ok=True)
            import uuid as _uuid
            local_path = os.path.join(tmp_dir, f"{_uuid.uuid4().hex[:12]}.png")
            with open(local_path, "wb") as f:
                f.write(img_resp.content)
            _t2i_log.info("图片已下载: %s (%.1fKB)", local_path, len(img_resp.content) / 1024)

            return f"[GENERATED_IMAGE]{local_path}[/GENERATED_IMAGE]\n图片已成功生成并保存。"
        except httpx.HTTPStatusError as e:
            body = e.response.text
            return f"文生图失败 (HTTP {e.response.status_code}): {body}"
        except Exception as e:
            return f"文生图失败: {e}"
    return text_to_image


def _make_describe_image():
    """图生文（视觉模型描述图片）。仅当 ENABLE_IMAGE_TO_TEXT 为真时加入。"""
    import logging as _logging
    _i2t_log = _logging.getLogger("agent.i2t")

    def _compress_image_to_base64(file_path: Path, max_side: int = 1280, quality: int = 85) -> tuple[str, str]:
        """将图片缩放+压缩为 JPEG base64，避免请求体过大导致上传超时。"""
        from PIL import Image
        import io, base64
        img = Image.open(file_path)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()
        _i2t_log.info("图片压缩: %s (%dx%d -> %dx%d, base64 %.1fKB)",
                       file_path.name, w, h, img.size[0], img.size[1], len(b64) / 1024)
        return b64, "image/jpeg"

    @tool
    def describe_image(image_path: str) -> str:
        """根据图片路径描述图片内容（图生文）。支持本地路径。
        当用户上传或提供图片路径并希望得到描述时使用。参数: image_path 图片本地路径。"""
        try:
            from agent.config import VL_API_KEY, VISION_MODEL
            path = Path(image_path).resolve()
            if not path.exists() or not path.is_file():
                return f"文件不存在或不是文件: {image_path}"
            b64, mime = _compress_image_to_base64(path)
            resp = httpx.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={"Authorization": f"Bearer {VL_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": VISION_MODEL,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            {"type": "text", "text": "请详细描述这张图片的内容。如果图中有文字，请完整提取出来。"},
                        ],
                    }],
                },
                timeout=httpx.Timeout(connect=30.0, write=300.0, read=180.0, pool=30.0),
            )
            if resp.status_code != 200:
                _i2t_log.error("图生文 HTTP %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"图生文失败: {e}"
    return describe_image


ALL_TOOLS = [get_current_datetime, calculate, web_search, remember_user_fact, get_weather]
EXTRA_TOOLS = [unit_convert, fetch_webpage, format_json, summarize_text, translate_text]

_cached_t2i = None
_cached_i2t = None


def get_all_tools(enable_t2i: bool | None = None, enable_i2t: bool | None = None):
    """返回当前应启用的工具列表。

    Args:
        enable_t2i: 是否启用文生图（None 表示跟随 config）
        enable_i2t: 是否启用图生文（None 表示跟随 config）
    """
    global _cached_t2i, _cached_i2t
    from agent.config import ENABLE_TEXT_TO_IMAGE, ENABLE_IMAGE_TO_TEXT

    t2i = enable_t2i if enable_t2i is not None else ENABLE_TEXT_TO_IMAGE
    i2t = enable_i2t if enable_i2t is not None else ENABLE_IMAGE_TO_TEXT

    tools = list(ALL_TOOLS) + list(EXTRA_TOOLS)
    if t2i:
        if _cached_t2i is None:
            _cached_t2i = _make_text_to_image()
        tools.append(_cached_t2i)
    if i2t:
        if _cached_i2t is None:
            _cached_i2t = _make_describe_image()
        tools.append(_cached_i2t)
    return tools

