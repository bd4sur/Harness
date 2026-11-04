from __future__ import annotations
import os, sys, readline

# ==========================================
# src/llm.py
# 统一 LLM API —— 把 OpenAI Completions SSE 响应解析成四种事件流。只支持 OpenAI 兼容格式。
# ==========================================
import json, codecs, urllib.request, urllib.error
from typing import Any, Generator, Optional
from dataclasses import dataclass

@dataclass
class Model:
    api_key: str
    model: str
    base_url: str = 'http://192.168.10.50:8888/v1'
    max_tokens: Optional[int] = None

# 思考内容（reasoning_content）回传策略。DeepSeek / Kimi 的 thinking 模型在多轮对话尤其
# 工具续写时要求把上一次的 reasoning_content 原样放回 assistant 消息，否则报 400 或推理链断裂。
# off：不回传；tool_calls：只在发起工具调用的那条 assistant 上回传（默认，token 最省）；all：全部回传。
# 回传内容永远追加在历史末尾，不修改已有前缀，因此不会破坏前缀缓存。
REASONING_ECHO = os.environ.get('HARNESS_REASONING_ECHO', 'tool_calls')

def context_to_openai_messages(context: dict) -> list[dict]:
    messages = []
    if context.get('system_prompt'):
        messages.append({'role': 'system', 'content': context['system_prompt']})

    for msg in context.get('messages', []):
        role = msg['role']
        content = msg['content']

        if role == 'user':
            messages.append({'role': 'user', 'content': content})
        elif role == 'assistant':
            # 内部格式：content 是字符串，tool_calls 在独立字段（{id, name, args}）
            text = content
            raw_tool_calls = msg.get('tool_calls') or []
            if not raw_tool_calls and not text:      # 空 assistant 消息会被 Kimi 判非法，见 AGENTS.md
                continue
            out = {'role': 'assistant', 'content': '' if raw_tool_calls else text}
            if raw_tool_calls:
                # 内部 tool_calls 转成 OpenAI 标准格式
                out['tool_calls'] = [{
                    'id': tc['id'], 'type': 'function',
                    'function': {'name': tc['name'], 'arguments': json.dumps(tc.get('args', {}), ensure_ascii=False)}
                } for tc in raw_tool_calls]
            # thinking 模型需要把上一轮的思考过程原样带回去（见 REASONING_ECHO）
            if msg.get('reasoning_content'):
                out['reasoning_content'] = msg['reasoning_content']
            messages.append(out)
        else:  # role == 'tool'：content 是 tool_result 块列表
            for b in content:
                messages.append({'role': 'tool', 'tool_call_id': b['tool_use_id'], 'content': b['content']})
    return messages

def stream(model: Model, context: dict, tools: Optional[list[dict]] = None, signal: Any = None) -> Generator[dict, None, None]:
    url = f"{model.base_url}/chat/completions"
    body: dict[str, Any] = {'model': model.model, 'stream': True, 'messages': context_to_openai_messages(context), 'stream_options': {'include_usage': True}}
    if model.max_tokens: body['max_tokens'] = model.max_tokens
    if tools: body['tools'] = [{'type': 'function', 'function': t} for t in tools]

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {model.api_key}'}
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers=headers, method='POST')

    try:
        with urllib.request.urlopen(req) as response:
            tool_call_buffers: dict[int, dict[str, str]] = {}
            stop_reason = 'end_turn'
            buffer = ""
            decoder = codecs.getincrementaldecoder('utf-8')('ignore')
            
            while True:
                # 用户中断：直接收摊，不再把缓冲区里读了一半的 tool_calls 当完整调用抛出去
                if signal and signal.aborted:
                    yield {'type': 'done', 'stop_reason': 'aborted'}
                    return

                # 优先消费 buffer 中已有的完整 SSE 行，每次只处理一条。
                # 原来 `while '\n' in buffer` 会一次性把 read(2048) 读到的多个事件
                # 全部解析并连续 yield，导致 TUI 在几毫秒内被调用多次，输出看起来
                # 就是“一大块”。改为每次只 yield 一个事件，让上层控制节奏。
                if '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                else:
                    # 读更小的块，避免一次吞掉多个 SSE 事件。
                    chunk = response.read(64)
                    if not chunk:
                        # 流结束，flush 解码器中可能残留的字节
                        final = decoder.decode(b'', final=True)
                        if final: buffer += final
                        break
                    # 增量解码：自动处理多字节字符边界，不完整的字节会缓存在 decoder 内部
                    buffer += decoder.decode(chunk)
                    continue

                line = line.strip()
                if not line.startswith('data: '): continue
                data = line[6:]
                if data == '[DONE]': break

                try: chunk_json = json.loads(data)
                except json.JSONDecodeError: continue

                # usage 通常在末尾单独一个 choices 为空的 chunk 里（stream_options.include_usage），
                # 拿它算前缀缓存命中率：缓存只对“与上次请求相同的前缀”生效。
                if chunk_json.get('usage'):
                    yield {'type': 'usage', 'usage': chunk_json['usage']}

                choice = (chunk_json.get('choices') or [{}])[0]
                if not choice: continue

                delta = choice.get('delta', {})
                if 'content' in delta and delta['content']:
                    # 把 content delta 拆成单个字符逐个 yield，提高 TUI 回调频率，
                    # 避免一个 delta 里包含多个词时一次性整段输出。
                    for ch in delta['content']:
                        yield {'type': 'text_delta', 'delta': ch}

                reasoning = delta.get('reasoning_content') or delta.get('reasoning')
                if reasoning:
                    for ch in reasoning:
                        yield {'type': 'reasoning_delta', 'delta': ch}

                if 'tool_calls' in delta and delta['tool_calls']:
                    for tc in delta['tool_calls']:
                        idx = tc.get('index', 0)
                        if idx not in tool_call_buffers:
                            tool_call_buffers[idx] = {'id': tc.get('id', f'call_{idx}'), 'name': '', 'args_buf': ''}
                        entry = tool_call_buffers[idx]
                        if 'id' in tc and tc['id']: entry['id'] = tc['id']
                        func = tc.get('function', {})
                        if 'name' in func and func['name']: entry['name'] = func['name']
                        if 'arguments' in func and func['arguments']: entry['args_buf'] += func['arguments']

                fr = choice.get('finish_reason')
                if fr in ('tool_calls', 'function_call'): stop_reason = 'tool_use'
                elif fr == 'length': stop_reason = 'max_tokens'
                    
    except KeyboardInterrupt:
        if signal: signal.aborted = True
        yield {'type': 'done', 'stop_reason': 'aborted'}
        return
    except Exception as e:
        if signal and signal.aborted:
            yield {'type': 'done', 'stop_reason': 'aborted'}
            return
        # HTTP 错误（如 400）时，原样打印服务端返回的错误响应体
        detail = ''
        if isinstance(e, urllib.error.HTTPError):
            try: detail = e.read().decode('utf-8', 'ignore')
            except Exception: detail = str(e)
            print(f'\n[http response error] {detail}')
        yield {'type': 'error', 'error': f'HTTP {e.code}: {detail}' if detail else str(e)}
        return

    for idx in sorted(tool_call_buffers.keys()):
        tc = tool_call_buffers[idx]
        args = {}
        if tc['args_buf']:
            try: args = json.loads(tc['args_buf'])
            except: args = {}
        yield {'type': 'tool_call', 'id': tc['id'], 'name': tc['name'], 'args': args}

    yield {'type': 'done', 'stop_reason': 'aborted' if (signal and signal.aborted) else stop_reason}

def build_assistant_message(text: str, tool_calls: list[dict], reasoning: str = '') -> Optional[dict]:
    text = text or ''
    if not text and not tool_calls:      # 空转轮次不写进历史，见 AGENTS.md「已知坑」
        return None
    msg = {'role': 'assistant', 'content': text}
    if tool_calls:
        msg['tool_calls'] = tool_calls
    if reasoning and (REASONING_ECHO == 'all' or (REASONING_ECHO == 'tool_calls' and tool_calls)):
        msg['reasoning_content'] = reasoning
    return msg

def cached_tokens(usage: dict) -> int:
    """从 usage 里取“命中前缀缓存的 prompt token 数”，兼容 OpenAI / DeepSeek 两种字段。"""
    details = usage.get('prompt_tokens_details') or {}
    return details.get('cached_tokens') or usage.get('prompt_cache_hit_tokens') or 0

def build_tool_result_message(results: list[dict]) -> dict:
    return {'role': 'tool', 'content': [{'type': 'tool_result', 'tool_use_id': r['tool_use_id'], 'content': r['content']} for r in results]}


# ==========================================
# src/agent.py
# Agent Loop —— 整个项目的灵魂：一个 while 循环。
# ==========================================
import json
from typing import Any, Generator, Callable
from dataclasses import dataclass

@dataclass
class AgentTool:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict, Any], str]

COMPACT_THRESHOLD = 50
KEEP_RECENT = 20

def compact_context(model: Model, context: dict, signal: Any = None, before_request: Any = None):
    if (signal and signal.aborted) or len(context.get('messages', [])) < COMPACT_THRESHOLD:
        return

    old_msgs = context['messages'][:-KEEP_RECENT]
    recent_msgs = context['messages'][-KEEP_RECENT:]

    # 摘要可能连发起 tool 的 assistant 一起摘掉，保留段以 tool 开头就缺了它（见 AGENTS.md）
    while recent_msgs and recent_msgs[0].get('role') == 'tool':
        idx = len(old_msgs) - 1
        while idx >= 0 and old_msgs[idx].get('role') != 'assistant':
            idx -= 1
        if idx < 0:
            # 找不到对应的 assistant，丢弃这条孤立的 tool 消息
            recent_msgs = recent_msgs[1:]
            continue
        recent_msgs = old_msgs[idx:] + recent_msgs
        old_msgs = old_msgs[:idx]
        break

    conv_text = '\n'.join([f"{m['role']}: {json.dumps(m['content'], ensure_ascii=False) if isinstance(m['content'], list) else m['content']}" for m in old_msgs])

    summary_ctx = {
        'system_prompt': '请将以下对话总结为简洁的上下文摘要，保留关键决策、已做的工作和待办事项。',
        'messages': [{'role': 'user', 'content': conv_text}]
    }

    summary, failed = '', False
    # 发请求前先让上层把当前上下文刷到磁盘（压缩前的状态同样需要在 session 里可见）
    if before_request: before_request(context)
    for ev in stream(model, summary_ctx, signal=signal):
        if ev['type'] == 'text_delta': summary += ev['delta']
        elif ev['type'] == 'error' or (ev['type'] == 'done' and ev['stop_reason'] == 'aborted'):
            failed = True; break

    if not failed and summary:
        new_messages = [{'role': 'user', 'content': f'[context summary]\n{summary}'}] + recent_msgs
        context['messages'] = new_messages
        # 压缩是一次“上下文重写”：早期消息被摘要取代，消息列表整体改变。
        # 返回详情，让调用方把重写后的完整上下文写进 session，
        # 否则 jsonl 会与真正发给模型的 messages 脱节（增量追加也无法表达删除）。
        return {
            'dropped': len(old_msgs),
            'kept': len(recent_msgs),
            'messages': [dict(m) for m in new_messages],
            'summary_request': {'system_prompt': summary_ctx['system_prompt'],
                                'messages': [dict(summary_ctx['messages'][0])]},
        }
    return None

def _append_assistant(context: dict, text: str, tool_calls: list[dict], reasoning: str = ''):
    """构建 assistant 消息并追加进 context；空消息（既无文本也无 tool_calls）自动跳过。"""
    msg = build_assistant_message(text, tool_calls, reasoning)
    if msg: context['messages'].append(msg)

def run_agent(model: Model, context: dict, tools: list[AgentTool], signal: Any = None, before_request: Any = None) -> Generator[dict, None, None]:
    """before_request：每次真正调用模型前回调一次，用于把上下文落盘（保证 jsonl 不缺内容）。"""
    tool_map = {t.name: t for t in tools}
    tool_defs = [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools]

    while True:
        compact_info = compact_context(model, context, signal, before_request)
        if compact_info:
            # 把“上下文被重写”这件事抛给上层，由上层原样落盘到 session。
            yield {'type': 'context_compacted', **compact_info}
        text, reasoning, stop_reason, tool_calls = '', '', 'end_turn', []

        if before_request: before_request(context)   # 发请求前先把上下文写入 session
        for ev in stream(model, context, tools=tool_defs, signal=signal):
            if ev['type'] == 'text_delta':
                text += ev['delta']
                yield {'type': 'assistant_text', 'delta': ev['delta']}
            elif ev['type'] == 'reasoning_delta':
                reasoning += ev['delta']            # 累起来，用于回传给 thinking 模型
                yield {'type': 'reasoning_text', 'delta': ev['delta']}
            elif ev['type'] == 'usage':
                yield {'type': 'usage', 'usage': ev['usage']}
            elif ev['type'] == 'tool_call':
                tool_calls.append({'id': ev['id'], 'name': ev['name'], 'args': ev['args']})
                yield {'type': 'tool_call', 'id': ev['id'], 'name': ev['name'], 'args': ev['args']}
            elif ev['type'] == 'done':
                stop_reason = ev['stop_reason']
                if stop_reason == 'aborted':
                    _append_assistant(context, text, [], reasoning)
                    yield {'type': 'turn_end', 'stop_reason': 'aborted'}
                    return
            elif ev['type'] == 'error':
                _append_assistant(context, text, [], reasoning)
                yield {'type': 'assistant_text', 'delta': f"\n[error] {ev['error']}"}
                yield {'type': 'turn_end', 'stop_reason': 'error'}
                return

        _append_assistant(context, text, tool_calls, reasoning)

        if not tool_calls:
            yield {'type': 'turn_end', 'stop_reason': 'end_turn' if stop_reason == 'tool_use' else stop_reason}
            return

        # 工具结果统一在这里构造：max_tokens 截断与正常执行只差 content 从哪来，
        # 收集完 results 后只 append 一次、只上报一遍，不再分两条路径。
        truncated = stop_reason == 'max_tokens'
        results = []
        for tc in tool_calls:
            if truncated:
                res = f'error: output truncated by max_tokens, tool "{tc["name"]}" args may be incomplete.'
            else:
                tool = tool_map.get(tc['name'])
                if not tool: res = f'error: tool "{tc["name"]}" not found'
                else:
                    try: res = tool.execute(tc['args'], signal)
                    except KeyboardInterrupt:
                        if signal: signal.aborted = True
                        res = 'error: aborted'
                    except Exception as e: res = f'error: {str(e)}'

            results.append({'tool_use_id': tc['id'], 'content': res})
            yield {'type': 'tool_result', 'id': tc['id'], 'name': tc['name'], 'result': res}
            if signal and signal.aborted: break

        # 中断时补上未执行的 tool_call，保证 tool_calls 与 tool 结果一一对应
        for tc in tool_calls[len(results):]:
            results.append({'tool_use_id': tc['id'], 'content': 'error: aborted'})
            yield {'type': 'tool_result', 'id': tc['id'], 'name': tc['name'], 'result': 'error: aborted'}

        context['messages'].append(build_tool_result_message(results))
        if signal and signal.aborted:
            yield {'type': 'turn_end', 'stop_reason': 'aborted'}
            return


# ==========================================
# src/tools.py
# 4 个内置工具 —— 能读写改代码并执行验证的最小集。
# ==========================================
import os, hashlib, subprocess, tempfile
from typing import Any

MAX_OUTPUT_LINES = 5000

def truncate_output(content: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    lines = content.split('\n')
    if len(lines) <= max_lines: return content
    kept = '\n'.join(lines[-max_lines:])
    # 用内容哈希命名，而不是 pid/计数器：同样的输出必须得到同样的工具结果字符串，
    # 否则工具结果不确定，跨会话的前缀缓存永远无法复用。
    digest = hashlib.sha1(content.encode('utf-8')).hexdigest()[:12]
    tmp_path = os.path.join(tempfile.gettempdir(), f'harness-output-{digest}.txt')
    with open(tmp_path, 'w', encoding='utf-8') as f: f.write(content)
    return f'[output truncated: showing last {max_lines} of {len(lines)} lines. full output: {tmp_path}]\n{kept}'

def _read_file(args: dict, signal: Any = None) -> str:
    with open(args['path'], 'r', encoding='utf-8') as f: content = f.read()
    return truncate_output(content)

def _write_file(args: dict, signal: Any = None) -> str:
    path, content = args['path'], args['content']
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: f.write(content)
    return f'wrote {path} ({len(content)} chars)'

def _edit(args: dict, signal: Any = None) -> str:
    path, old_str, new_str = args['path'], args['old_string'], args['new_string']
    with open(path, 'r', encoding='utf-8') as f: content = f.read()
    count = content.count(old_str)
    if count == 0: raise ValueError(f'old_string not found in {path}')
    if count > 1: raise ValueError(f'old_string matches {count} places in {path}, must be unique')
    with open(path, 'w', encoding='utf-8') as f: f.write(content.replace(old_str, new_str, 1))
    return f'edited {path}: replaced {len(old_str)} chars'

def _run_bash(args: dict, signal: Any = None) -> str:
    cmd = args['command']
    # 强制用户确认，直接回车确认，Ctrl+C 取消
    try:
        confirm = input(f'\033[1;31m[run_bash confirm] {cmd}\033[0m\n\033[1;31m执行该命令？直接回车确认，Ctrl+C 取消\033[0m >>> ')
    except KeyboardInterrupt:
        if signal: signal.aborted = True
        return 'aborted by user'
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        output = f'[stderr] {res.stderr}\n[stdout] {res.stdout}' if res.stderr else res.stdout
        return truncate_output(output)
    except subprocess.TimeoutExpired: return 'error: command timed out'
    except KeyboardInterrupt:
        if signal: signal.aborted = True
        return 'aborted'
    except Exception as e: return f'[error] {str(e)}'

TOOL_SPECS = [      # (name, description, parameters, execute) —— 一个工具一条
    ('read_file', '读取文件内容。参数：path（文件路径）。超长文件只返回末尾 5000 行，并给出完整内容的临时文件路径。',
     {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}, _read_file),
    ('write_file', '写入文件（覆盖）。参数：path（路径）、content（内容）',
     {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']}, _write_file),
    ('edit', '编辑文件：精确替换一段文本。参数：path、old_string、new_string。old_string 必须在文件中唯一匹配，否则报错。',
     {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_string': {'type': 'string'}, 'new_string': {'type': 'string'}}, 'required': ['path', 'old_string', 'new_string']}, _edit),
    ('run_bash', '执行 shell 命令。参数：command（命令字符串）。返回 stdout+stderr，超长输出只保留末尾 5000 行并给出临时文件路径。',
     {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}, _run_bash),
]

def builtin_tools() -> list[AgentTool]:
    return [AgentTool(*spec) for spec in TOOL_SPECS]


# ==========================================
# src/tui.py
# 极简终端界面 —— 单行输入 + 流式输出 + Ctrl+C 打断。
# ==========================================
import sys, json

class Tui:
    # ANSI 样式：斜体青色用于思考过程，黄色用于工具调用信息，重置用于恢复默认。
    REASONING_STYLE = '\033[3;36m'
    TOOL_STYLE = '\033[1;33m'
    NOTICE_STYLE = '\033[2m'
    RESET = '\033[0m'
    # 单行渲染表：kind → (渲染函数, 样式)。text / result / turn_end / notice / tool_call
    # 一律走 print()，不再每种写一个只做转发的 def。
    LINE = {
        'text':      (lambda s: s, ''),
        'result':    (lambda name: f'[result: {name}] Finished\n', TOOL_STYLE),
        'turn_end':  (lambda: '\n', ''),
        'notice':    (lambda s: f'{s}\n', NOTICE_STYLE),
        'tool_call': (lambda name, args: f'\n[tool: {name}] {json.dumps(args, ensure_ascii=False)}\n', TOOL_STYLE),
    }

    def __init__(self):
        self.in_reasoning = False

    def _end_reasoning(self):
        if self.in_reasoning:
            sys.stdout.write(f']\n{self.RESET}')
            self.in_reasoning = False

    def print(self, kind: str, *args):
        """非思考输出的唯一出口：先关掉思考块，再按 LINE 表渲染并 flush。"""
        self._end_reasoning()
        render, style = self.LINE[kind]
        sys.stdout.write(f'{style}{render(*args)}{self.RESET if style else ""}')
        sys.stdout.flush()

    def print_reasoning(self, delta: str):
        # 思考块内部不走 print（那会提前关掉思考块），风格在开头挂一次
        if not self.in_reasoning:
            sys.stdout.write(f'{self.REASONING_STYLE}[Thinking: ')
            self.in_reasoning = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    def print_usage(self, usage: dict):
        """一行缓存命中率：命中数 / prompt 总数，便于判断前缀缓存有没有生效。"""
        prompt = usage.get('prompt_tokens') or 0
        hit = cached_tokens(usage)
        ratio = f'{hit * 100 // prompt}%' if prompt else '-'
        self.print('notice', f"[cache {ratio}] {hit}/{prompt} prompt tokens reused")


# ==========================================
# src/cli.py
# 拼装层 —— 把 llm / agent / tui / tools 粘起来，是唯一入口。
# ==========================================
import os, sys, json, time
from datetime import datetime

SESSION_DIR = os.path.join(os.getcwd(), 'sessions')

def _new_session_file() -> str:
    return os.path.join(SESSION_DIR, f'session_{int(time.time() * 1000)}.jsonl')

def _pick_session_file() -> str:
    """HARNESS_SESSION 显式指定 > sessions/ 里最新的一份（续接）> 新建时间戳文件；=none 强制新开。"""
    explicit = os.environ.get('HARNESS_SESSION', '')
    if explicit: return _new_session_file() if explicit == 'none' else explicit
    if os.path.isdir(SESSION_DIR):
        history = sorted((f for f in os.listdir(SESSION_DIR) if f.startswith('session_') and f.endswith('.jsonl')),
                         key=lambda f: os.path.getmtime(os.path.join(SESSION_DIR, f)))
        if history: return os.path.join(SESSION_DIR, history[-1])
    return _new_session_file()

SESSION_FILE = _pick_session_file()
# 注意：系统提示词必须是常量，不能含时间、路径、主机名等每次启动都会变的内容。
# 服务端前缀缓存按“逐 token 相同的前缀”命中，头部一变整段缓存全废；
# 需要环境信息时让模型自己调 run_bash 去查（见下方“环境信息”）。
SYSTEM_PROMPT = """你是一个严谨的本地编码助手。

你的目标：通过工具与用户协作，高质量完成代码相关任务。

环境信息：
你不掌握任何运行环境的先验事实（当前时间、工作目录、操作系统版本等），
需要时一律用 run_bash 现查，不要拍脑袋假设：
- 当前时间 / 时区：date、date -Iseconds
- 当前工作目录：pwd
- 操作系统 / 架构：uname -a、cat /etc/os-release
- 用户 / 主机名：whoami、hostname
- Python 解释器版本：python3 -V、which python3
- 本程序的模型配置：echo $HARNESS_MODEL $HARNESS_BASE_URL $HARNESS_MAX_TOKENS

核心工作流（必须遵守）：
1. 先读：修改任何文件前，先用 read_file 读取目标文件和相关上下文。
2. 再计划：简要说明你要做什么、分几步、每步改什么。
3. 最小修改：优先使用 edit 做精确文本替换；只有新建文件或必须整体重写时，才使用 write_file。
4. 后验证：修改完成后，必须运行相关命令（测试、编译、lint、运行脚本等）验证结果。
5. 汇报：用一句话总结本回合做了什么、验证结果如何。

工具使用规范：
- read_file：读取文件内容，大文件会自动截断末尾若干行，并给出完整输出的临时文件路径。
- edit：old_string 必须在文件中唯一存在，替换前请在心里核对 old_string 与文件内容是否完全匹配。
- write_file：会覆盖原文件，使用前先确认目标不存在或用户明确要求覆盖。
- run_bash：执行 shell 命令，执行前工具会要求用户确认；请一次性给出正确的命令，避免交互式命令（如 vim、top）。

错误处理：
- 如果 edit 报错 old_string 不匹配，先重新 read_file 确认内容，再修正 old_string 重试。
- 如果 run_bash 返回错误，先分析 stderr，判断是命令问题、环境问题还是预期内错误，再决定下一步。
- 不要连续多次用同样的参数重试同一工具。

输出风格：
- 对用户的回复简洁、结构化，避免长篇大论。
- 思考过程可以用 [Thinking: ...] 展示，但对外输出要聚焦在行动和结果上。
- 每轮结束时，用一句话总结当前进度。

安全边界：
- 不主动删除用户文件。
- 不执行可能破坏工作目录的命令（rm -rf、重定向覆盖关键文件等）。
- 需要覆盖或删除时，先征得用户同意。
"""

class Signal:
    def __init__(self): self.aborted = False

def load_session(path: str = SESSION_FILE) -> list[dict]:
    """按记录类型折叠 session jsonl 重建 messages：消息追加、snapshot 整份替换、残缺尾巴丢弃。"""
    if not os.path.exists(path): return []
    messages: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            if not isinstance(rec, dict): continue
            if 'role' in rec:                      # 普通消息记录
                messages.append(rec)
            elif rec.get('type') == 'snapshot':    # 上下文重写点：整份替换
                messages = rec.get('messages', [])
            # meta / system_prompt / usage 等记录仅用于审计，不参与上下文重建
    # 崩溃/强杀可能把文件停在“assistant 已发起 tool_calls、结果还没落盘”的位置；
    # 这种残缺尾巴会让 OpenAI 兼容 API 报 400，续接时把 tool 结果连同发起者一起丢弃。
    while messages and (messages[-1].get('role') == 'tool'
                        or (messages[-1].get('role') == 'assistant' and messages[-1].get('tool_calls'))):
        messages.pop()
    return messages

# session jsonl 记录类型（每行一个 JSON 对象）：
#   {"role": ...}                                   普通消息，与发给模型的 message 一一对应
#   {"type": "meta", ...}                            请求的固定部分：模型名 / base_url / max_tokens / 工具 schema
#   {"type": "system_prompt", "content": ...}       完整系统提示词（启动时写一份全文）
#   {"type": "snapshot", "messages": [...]}         上下文被重写（压缩）后的完整消息列表
#   {"type": "snapshot", "summary_request": ...}    压缩时额外发给模型的摘要请求，也一并留痕
#   {"type": "usage", ...}                          每次请求的 token 用量与缓存命中
# 原则：凡是发给模型的上下文（system、messages、tools、max_tokens）都必须能在本文件里找到，
#      按行读回的顺序即真实上下文顺序。

class SessionLog:
    """负责把内存里的 context 完整、增量地写进 session jsonl。"""

    def __init__(self, path: str, restored: Optional[list[dict]] = None):
        self.path = path
        # context['messages'] 中已落盘的前缀长度。续接时先扣掉恢复出来的部分，
        # 否则第一次 sync 会把已落盘的历史再追加一遍，下次续接就读重了。
        self.message_cursor = len(restored or [])

    def append(self, record: dict):
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def start(self, model: Model, tools: list[AgentTool], context: dict):
        """刚启动时：先落盘请求的固定部分与系统提示词，再发任何请求。"""
        self.append({
            'type': 'meta',
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': model.model,
            'base_url': model.base_url,
            'max_tokens': model.max_tokens,
            'stream': True,
            'tools': [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools],
        })
        self.append({'type': 'system_prompt', 'content': context['system_prompt']})

    def sync(self, context: dict):
        """把新增的消息追加到文件；若发现上下文被意外重写则整份快照。"""
        messages = context['messages']
        if len(messages) < self.message_cursor:
            self.append({'type': 'snapshot', 'reason': 'rewritten', 'messages': messages})
            self.message_cursor = len(messages)
            return
        for msg in messages[self.message_cursor:]:
            self.append(msg)
        self.message_cursor = len(messages)

    def on_compaction(self, ev: dict):
        """上下文压缩：写完整快照 + 摘要请求全文，重置增量游标。"""
        self.append({
            'type': 'snapshot',
            'reason': 'compaction',
            'dropped': ev['dropped'],
            'kept': ev['kept'],
            'summary_request': ev['summary_request'],
            'messages': ev['messages'],
        })
        self.message_cursor = len(ev['messages'])

def main():
    api_key = os.environ.get('HARNESS_API_KEY')
    if not api_key:
        print('未设置 HARNESS_API_KEY 环境变量，默认为空（通过 export HARNESS_API_KEY=xxx 设置）', file=sys.stderr)
        api_key = "sk-0"

    model = Model(
        api_key=api_key,
        model=os.environ.get('HARNESS_MODEL', 'deepseek-v4-flash-0731'),
        base_url=os.environ.get('HARNESS_BASE_URL', 'http://192.168.10.50:8888/v1'),
        max_tokens=int(os.environ.get('HARNESS_MAX_TOKENS', '65536'))
    )

    restored_messages = load_session()
    context = {'system_prompt': SYSTEM_PROMPT, 'messages': restored_messages}
    tools_list = builtin_tools()
    ui = Tui()
    log = SessionLog(SESSION_FILE, restored_messages)
    log.start(model, tools_list, context)   # 先把 meta / system prompt 落盘，再发任何请求
    flush_context = log.sync                # 每次调用模型前先落盘，保证 jsonl 不缺上下文

    if restored_messages: print(f"续接会话 {SESSION_FILE}（恢复 {len(restored_messages)} 条消息）")
    else: print(f"新建会话 {SESSION_FILE}")
    print("Agent 已启动 (输入 Ctrl+C 可随时中断当前任务，连续两次 Ctrl+C 退出程序)")
    print("多行输入以空行结束；空行本身不会保留。")
    while True:
        try:
            first_line = input('> ').rstrip('\n')
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break

        if not first_line.strip():
            continue

        # 多行输入：空行结束。后续行用 ': ' 提示。
        lines = [first_line]
        while True:
            try:
                line = input(': ').rstrip('\n')
            except (EOFError, KeyboardInterrupt):
                break
            if not line.strip():
                break
            lines.append(line)

        text = '\n'.join(lines).strip()
        if not text:
            continue

        context['messages'].append({'role': 'user', 'content': text})
        log.sync(context)   # 先把 user 消息落盘，避免中途崩掉时上下文缺尾巴

        signal = Signal()
        try:
            for ev in run_agent(model, context, tools_list, signal, before_request=flush_context):
                if ev['type'] == 'assistant_text': ui.print('text', ev['delta'])
                elif ev['type'] == 'reasoning_text': ui.print_reasoning(ev['delta'])
                elif ev['type'] == 'tool_call': ui.print('tool_call', ev['name'], ev['args'])
                elif ev['type'] == 'tool_result': ui.print('result', ev['name'])
                elif ev['type'] == 'context_compacted':
                    # 压缩重写了 messages，立刻写一份完整快照（含摘要请求原文）
                    log.on_compaction(ev)
                    ui.print('notice', f"[context compacted] dropped {ev['dropped']} messages, kept {ev['kept']}")
                elif ev['type'] == 'usage':
                    # 缓存观测：命中率接近 0 说明前缀被破坏（典型：系统提示词含时间戳、刚压缩完）
                    usage = ev['usage']
                    ui.print_usage(usage)
                    log.append({'type': 'usage', 'prompt_tokens': usage.get('prompt_tokens'),
                                'cached_tokens': cached_tokens(usage),
                                'completion_tokens': usage.get('completion_tokens')})
                elif ev['type'] == 'turn_end':
                    if ev['stop_reason'] == 'max_tokens': ui.print('text', '\n[output truncated by max_tokens]')
                    if ev['stop_reason'] == 'error': ui.print('text', '\n[error occurred]')
                    ui.print('turn_end')
        except KeyboardInterrupt:
            signal.aborted = True
            print('\n[aborted by user]')

        # 本回合遗留的上下文（含中断 / 错误路径）再刷一遍，幂等
        log.sync(context)

if __name__ == '__main__':
    main()
