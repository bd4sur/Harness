from __future__ import annotations
import os
import sys
import readline

# ==========================================
# src/llm.py
# 统一 LLM API —— 把 OpenAI Completions SSE 响应解析成四种事件流。只支持 OpenAI 兼容格式。
# ==========================================
import json
import codecs
import urllib.request
import urllib.error
from typing import Any, Generator, Optional
from dataclasses import dataclass

@dataclass
class Model:
    api_key: str
    model: str
    base_url: str = 'http://192.168.10.50:8888/v1'
    max_tokens: Optional[int] = None

def context_to_openai_messages(context: dict) -> list[dict]:
    messages = []
    if context.get('system_prompt'):
        messages.append({'role': 'system', 'content': context['system_prompt']})

    for msg in context.get('messages', []):
        role = msg['role']
        content = msg['content']

        if isinstance(content, str) and role != 'assistant':
            messages.append({'role': role, 'content': content})
            continue

        if role == 'assistant':
            text = ''
            raw_tool_calls = []
            if isinstance(content, str):
                # 新格式：content 直接是字符串，tool_calls 在独立字段（内部格式）
                text = content
                raw_tool_calls = msg.get('tool_calls', [])
            else:
                # 兼容旧格式：content 是 parts 数组（text / tool_use）
                for b in content:
                    if b['type'] == 'text': text += b['text']
                    elif b['type'] == 'tool_use':
                        raw_tool_calls.append({
                            'id': b['id'], 'type': 'function',
                            'function': {'name': b['name'], 'arguments': json.dumps(b['input'], ensure_ascii=False)}
                        })
            # 修复：部分 OpenAI 兼容 API 要求 assistant 消息 content 不能为空，
            # 且带 tool_calls 时 content 必须为空字符串。
            # 修复：跳过完全空的 assistant 消息（无内容且无 tool_calls），避免某些
            # OpenAI 兼容 API（如 Kimi）将 content="" 且无 tool_calls 的消息视为无效。
            if not raw_tool_calls and not text:
                continue
            if raw_tool_calls:
                # 新格式的内部 tool_calls 需要转成 OpenAI 标准格式
                tool_calls = raw_tool_calls if raw_tool_calls[0].get('type') == 'function' else [
                    {
                        'id': tc['id'], 'type': 'function',
                        'function': {'name': tc['name'], 'arguments': json.dumps(tc.get('args', {}), ensure_ascii=False)}
                    } for tc in raw_tool_calls
                ]
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': tool_calls})
            else:
                messages.append({'role': 'assistant', 'content': text})
        elif role == 'tool':
            # 标准格式：tool result 以 role='tool' 的 block 列表保存
            for b in content:
                if b['type'] == 'tool_result':
                    messages.append({'role': 'tool', 'tool_call_id': b['tool_use_id'], 'content': b['content']})
        else:
            # 兼容旧格式：tool result 曾以 role='user' 的 block 列表保存
            for b in content:
                if b['type'] == 'tool_result':
                    messages.append({'role': 'tool', 'tool_call_id': b['tool_use_id'], 'content': b['content']})
                elif b['type'] == 'text':
                    messages.append({'role': 'user', 'content': b['text']})
    return messages

def stream(model: Model, context: dict, tools: Optional[list[dict]] = None, signal: Any = None) -> Generator[dict, None, None]:
    url = f"{model.base_url}/chat/completions"
    body: dict[str, Any] = {'model': model.model, 'stream': True, 'messages': context_to_openai_messages(context)}
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
                if signal and signal.aborted: break

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

                choice = chunk_json.get('choices', [{}])[0]
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
                if fr == 'tool_calls': stop_reason = 'tool_use'
                elif fr == 'length': stop_reason = 'max_tokens'
                    
    except KeyboardInterrupt:
        if signal: signal.aborted = True
        yield {'type': 'done', 'stop_reason': 'aborted'}
        return
    except urllib.error.HTTPError as e:
        # HTTP 错误（如 400）时，原样打印服务端返回的错误响应体
        try: err_body = e.read().decode('utf-8', 'ignore')
        except Exception: err_body = str(e)
        print(f'\n[http response error] {err_body}')
        yield {'type': 'error', 'error': f'HTTP {e.code}: {err_body}'}
        return
    except Exception as e:
        if signal and signal.aborted:
            yield {'type': 'done', 'stop_reason': 'aborted'}
            return
        yield {'type': 'error', 'error': str(e)}
        return

    for idx in sorted(tool_call_buffers.keys()):
        tc = tool_call_buffers[idx]
        args = {}
        if tc['args_buf']:
            try: args = json.loads(tc['args_buf'])
            except: args = {}
        yield {'type': 'tool_call', 'id': tc['id'], 'name': tc['name'], 'args': args}

    yield {'type': 'done', 'stop_reason': 'aborted' if (signal and signal.aborted) else stop_reason}

def build_assistant_message(text: str, tool_calls: list[dict]) -> Optional[dict]:
    # 修复：避免返回 content: []，统一使用简单结构。
    # 当没有文本且没有 tool_calls 时，返回 None，调用方应跳过这条空消息，
    # 避免某些 OpenAI 兼容 API（如 Kimi）把 content="" 且无 tool_calls 的
    # assistant 消息视为无效格式而报 400。
    text = text or ''
    if not text and not tool_calls:
        return None
    msg = {'role': 'assistant', 'content': text}
    if tool_calls:
        msg['tool_calls'] = tool_calls
    return msg

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

    # 修复：截断后如果 recent_msgs 以 tool 消息开头，其对应的 assistant/tool_calls
    # 可能被摘要到 user summary 中，导致 OpenAI 兼容 API 报 400（tool 消息前必须
    # 紧跟产生它的 assistant）。向前扩展，把对应的 assistant 一并包含进来。
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
            'summary_request': json.loads(json.dumps(summary_ctx, ensure_ascii=False)),
        }
    return None

def run_agent(model: Model, context: dict, tools: list[AgentTool], signal: Any = None, before_request: Any = None) -> Generator[dict, None, None]:
    """before_request：每次真正调用模型前回调一次，用于把上下文落盘（保证 jsonl 不缺内容）。"""
    tool_map = {t.name: t for t in tools}
    tool_defs = [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools]

    while True:
        compact_info = compact_context(model, context, signal, before_request)
        if compact_info:
            # 把“上下文被重写”这件事抛给上层，由上层原样落盘到 session。
            yield {'type': 'context_compacted', **compact_info}
        text, stop_reason, tool_calls = '', 'end_turn', []

        try:
            if before_request: before_request(context)   # 发请求前先把上下文写入 session
            for ev in stream(model, context, tools=tool_defs, signal=signal):
                if ev['type'] == 'text_delta':
                    text += ev['delta']
                    yield {'type': 'assistant_text', 'delta': ev['delta']}
                elif ev['type'] == 'reasoning_delta':
                    yield {'type': 'reasoning_text', 'delta': ev['delta']}
                elif ev['type'] == 'tool_call':
                    tool_calls.append({'id': ev['id'], 'name': ev['name'], 'args': ev['args']})
                    yield {'type': 'tool_call', 'id': ev['id'], 'name': ev['name'], 'args': ev['args']}
                elif ev['type'] == 'done':
                    stop_reason = ev['stop_reason']
                    if stop_reason == 'aborted':
                        msg = build_assistant_message(text, [])
                        if msg: context['messages'].append(msg)
                        yield {'type': 'turn_end', 'stop_reason': 'aborted'}
                        return
                elif ev['type'] == 'error':
                    msg = build_assistant_message(text, [])
                    if msg: context['messages'].append(msg)
                    yield {'type': 'assistant_text', 'delta': f"\n[error] {ev['error']}"}
                    yield {'type': 'turn_end', 'stop_reason': 'error'}
                    return
        except KeyboardInterrupt:
            if signal: signal.aborted = True
            msg = build_assistant_message(text, [])
            if msg: context['messages'].append(msg)
            yield {'type': 'turn_end', 'stop_reason': 'aborted'}
            return

        msg = build_assistant_message(text, tool_calls)
        if msg: context['messages'].append(msg)

        if stop_reason == 'max_tokens' and tool_calls:
            results = [{'tool_use_id': tc['id'], 'content': f'error: output truncated by max_tokens, tool "{tc["name"]}" args may be incomplete.'} for tc in tool_calls]
            context['messages'].append(build_tool_result_message(results))
            for i, tc in enumerate(tool_calls):
                yield {'type': 'tool_result', 'id': tc['id'], 'name': tc['name'], 'result': results[i]['content']}
            continue

        if not tool_calls:
            yield {'type': 'turn_end', 'stop_reason': 'end_turn' if stop_reason == 'tool_use' else stop_reason}
            return

        results = []
        for tc in tool_calls:
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
import os
import subprocess
import sys
import tempfile
from typing import Any

MAX_OUTPUT_LINES = 5000
_truncate_counter = 0

def truncate_output(content: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    lines = content.split('\n')
    if len(lines) <= max_lines: return content
    kept = '\n'.join(lines[-max_lines:])
    global _truncate_counter
    tmp_path = os.path.join(tempfile.gettempdir(), f'nanopi-output-{os.getpid()}-{_truncate_counter}.txt')
    _truncate_counter += 1
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

def builtin_tools() -> list[AgentTool]:
    return [
        AgentTool(
            'read_file',
            '读取文件内容。参数：path（文件路径）。大文件截取最后 200 行。',
            {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']},
            _read_file),
        AgentTool(
            'write_file',
            '写入文件（覆盖）。参数：path（路径）、content（内容）',
            {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}, 'required': ['path', 'content']},
            _write_file),
        AgentTool(
            'edit',
            '编辑文件：精确替换一段文本。参数：path、old_string、new_string。old_string 必须在文件中唯一匹配，否则报错。',
            {'type': 'object', 'properties': {'path': {'type': 'string'}, 'old_string': {'type': 'string'}, 'new_string': {'type': 'string'}}, 'required': ['path', 'old_string', 'new_string']},
            _edit),
        AgentTool(
            'run_bash',
            '执行 shell 命令。参数：command（命令字符串）。返回 stdout+stderr，截取最后 200 行。',
            {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']},
            _run_bash)
    ]


# ==========================================
# src/tui.py
# 极简终端界面 —— 单行输入 + 流式输出 + Ctrl+C 打断。
# ==========================================
import sys
import json

class Tui:
    # ANSI 样式：斜体青色用于思考过程，黄色用于工具调用信息，重置用于恢复默认。
    REASONING_STYLE = '\033[3;36m'
    TOOL_STYLE = '\033[1;33m'
    NOTICE_STYLE = '\033[2m'
    RESET = '\033[0m'

    def __init__(self):
        self.in_reasoning = False

    def _end_reasoning(self):
        if self.in_reasoning:
            sys.stdout.write(f']\n{self.RESET}')
            self.in_reasoning = False

    def print_text(self, delta: str):
        self._end_reasoning()
        sys.stdout.write(delta)
        sys.stdout.flush()

    def print_reasoning(self, delta: str):
        if not self.in_reasoning:
            sys.stdout.write(f'{self.REASONING_STYLE}[Thinking: ')
            self.in_reasoning = True
        sys.stdout.write(delta)
        sys.stdout.flush()

    def print_tool_call(self, name: str, args: dict):
        self._end_reasoning()
        sys.stdout.write(f'{self.TOOL_STYLE}\n[tool: {name}] {json.dumps(args, ensure_ascii=False)}\n{self.RESET}')
        sys.stdout.flush()

    def print_tool_result(self, name: str, result: str):
        self._end_reasoning()
        # sys.stdout.write(f'{self.TOOL_STYLE}[result: {name}] {result}\n{self.RESET}')
        sys.stdout.write(f'{self.TOOL_STYLE}[result: {name}] Finished\n{self.RESET}')
        sys.stdout.flush()

    def print_turn_end(self):
        self._end_reasoning()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def print_notice(self, text: str):
        self._end_reasoning()
        sys.stdout.write(f'{self.NOTICE_STYLE}{text}\n{self.RESET}')
        sys.stdout.flush()


# ==========================================
# src/commands.py
# 斜杠指令 —— 以 / 开头的输入在此本地解析，不会作为普通对话发给模型。
# 通过注册表扩展：新增内置指令只需向 builtin_commands() 注册一个 handler。
# 未注册的指令名默认当作 skill 名，读取 skills/<name>.md 拼进 system_prompt。
# ==========================================
import os
import re
from typing import Callable, Optional

SKILLS_DIR = os.path.join(os.getcwd(), 'skills')

def parse_command(text: str) -> Optional[tuple[str, str]]:
    """把 /name args 解析成 (name, args)；不是指令返回 None。"""
    if not text.startswith('/'):
        return None
    parts = text[1:].split(None, 1)
    name = parts[0].lower() if parts else ''
    args = parts[1] if len(parts) > 1 else ''
    return name, args

def _safe_skill_path(name: str) -> str:
    """把 skill 名规整为 skills 目录下的安全路径，防止路径逃逸。"""
    if not name: raise ValueError('skill 名为空')
    base = os.path.basename(name)
    if base != name or base in ('.', '..'):
        raise ValueError(f'非法 skill 名: {name}')
    return os.path.join(SKILLS_DIR, base + '.md')

def _strip_frontmatter(content: str) -> str:
    """去掉 markdown 顶部由 --- 包裹的 frontmatter。"""
    if content.startswith('---'):
        m = re.match(r'^---\s*\n.*?\n---\s*\n?', content, re.DOTALL)
        if m: return content[m.end():]
    return content

def _load_skill(name: str, context: dict) -> str:
    path = _safe_skill_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f'skill 不存在: {name} (期望 {path})')
    with open(path, 'r', encoding='utf-8') as f: content = f.read()
    skill_text = _strip_frontmatter(content).strip()
    context['system_prompt'] = context.get('system_prompt', '') + \
        f"\n\n---- skill: {name} ----\n{skill_text}\n---- end skill: {name} ----"
    return f'已加载 skill "{name}" 并拼接到 system prompt。'

def _cmd_help(args: str, context: dict) -> str:
    return '\n'.join([
        '可用斜杠指令（以 / 开头）：',
        '  /help         显示本帮助',
        '  /skills       列出 skills 目录下可用的 skill',
        '  /skill <name> 显式加载 skills/<name>.md 到 system prompt',
        '  /<name>       未注册指令默认当作 skill 名，等价于 /skill <name>',
    ])

def _cmd_skills(args: str, context: dict) -> str:
    if not os.path.isdir(SKILLS_DIR):
        return 'skills 目录不存在'
    names = sorted(f[:-3] for f in os.listdir(SKILLS_DIR) if f.endswith('.md'))
    return '可用 skills: ' + (', '.join(names) if names else '(无)')

def _cmd_skill(args: str, context: dict) -> str:
    name = args.strip()
    if not name:
        return '用法: /skill <name> —— 加载 skills/<name>.md'
    return _load_skill(name, context)

def builtin_commands() -> dict[str, Callable[[str, dict], str]]:
    return {
        'help': _cmd_help,
        'skills': _cmd_skills,
        'skill': _cmd_skill,
    }

def handle_command(text: str, context: dict) -> Optional[str]:
    """处理斜杠指令。是指令则执行并返回展示文本，否则返回 None。"""
    parsed = parse_command(text)
    if parsed is None: return None
    name, args = parsed
    if not name: return _cmd_help(args, context)     # 裸 "/" 显示帮助
    handlers = builtin_commands()
    if name in handlers:                      # 已注册的内置指令优先
        return handlers[name](args, context)
    try:                                      # 未注册的当作 skill 名
        return _load_skill(name, context)
    except FileNotFoundError:
        return f'未知指令或 skill 不存在: /{name}\n提示: 输入 /help 查看可用指令，/skills 查看可用 skills。'
    except ValueError as e:
        return str(e)


# ==========================================
# src/cli.py
# 拼装层 —— 把 llm / agent / tui / tools 粘起来，是唯一入口。
# ==========================================
import os
import sys
import json
import time
import getpass
import platform
import socket
from datetime import datetime

SESSION_DIR = os.path.join(os.getcwd(), 'sessions')
SESSION_FILE = os.path.join(SESSION_DIR, f'session_{int(time.time() * 1000)}.jsonl')
SYSTEM_PROMPT = f"""你是一个严谨的本地编码助手。

当前环境信息：
- 当前日期时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}（本地时间）
- 当前工作目录：{os.getcwd()}/
- 操作系统：{platform.system()} {platform.release()} ({platform.machine()})
- Python 版本：{platform.python_version()}
- 主机名：{socket.gethostname()}
- 当前用户：{getpass.getuser()}
- HARNESS_MODEL：{os.environ.get('HARNESS_MODEL', '未设置，使用默认值 deepseek-v4-flash-0731')}
- HARNESS_BASE_URL：{os.environ.get('HARNESS_BASE_URL', '未设置，使用默认值 http://192.168.10.50:8888/v1')}
- HARNESS_MAX_TOKENS：{os.environ.get('HARNESS_MAX_TOKENS', '未设置，使用默认值 1048576')}

你的目标：通过工具与用户协作，高质量完成代码相关任务。

核心工作流（必须遵守）：
1. 先读：修改任何文件前，先用 read_file 读取目标文件和相关上下文。
2. 再计划：简要说明你要做什么、分几步、每步改什么。
3. 最小修改：优先使用 edit 做精确文本替换；只有新建文件或必须整体重写时，才使用 write_file。
4. 后验证：修改完成后，必须运行相关命令（测试、编译、lint、运行脚本等）验证结果。
5. 汇报：用一句话总结本回合做了什么、验证结果如何。

工具使用规范：
- read_file：读取文件内容，大文件会自动截断最后若干行。需要更多上下文时可指定具体范围（但目前不支持行号，可读取后自己分析）。
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

def load_session() -> tuple[list[dict], Optional[str]]:
    """从 session jsonl 重建上下文。

    返回 (messages, system_prompt)。记录类型见下方 SessionLog 的说明：
    普通消息按顺序追加，snapshot 记录表示上下文重写（压缩）需要整份替换。
    """
    if not os.path.exists(SESSION_FILE): return [], None
    messages: list[dict] = []
    system_prompt = None
    with open(SESSION_FILE, 'r', encoding='utf-8') as f:
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
            elif rec.get('type') == 'system_prompt':
                system_prompt = rec.get('content')
            # meta 等记录仅用于审计，不参与上下文重建
    return messages, system_prompt

# session jsonl 记录类型（每行一个 JSON 对象）：
#   {"role": ...}                                   普通消息，与发给模型的 message 一一对应
#   {"type": "meta", ...}                            请求的固定部分：模型名 / base_url / max_tokens / 工具 schema
#   {"type": "system_prompt", "content": ...}       完整系统提示词；每次变化（如 /skill）都重新写一份全文
#   {"type": "snapshot", "messages": [...]}         上下文被重写（压缩）后的完整消息列表
#   {"type": "snapshot", "summary_request": ...}    压缩时额外发给模型的摘要请求，也一并留痕
# 原则：凡是发给模型的上下文（system、messages、tools、max_tokens）都必须能在本文件里找到，
#      按行读回的顺序即真实上下文顺序。

class SessionLog:
    """负责把内存里的 context 完整、增量地写进 session jsonl。"""

    def __init__(self, path: str):
        self.path = path
        self.message_cursor = 0      # context['messages'] 中已落盘的前缀长度
        self.system_prompt = None    # 已落盘的系统提示词

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
        self.sync_system_prompt(context)

    def sync_system_prompt(self, context: dict):
        """系统提示词有变动（初始化、/skill 追加）就写一份完整全文。"""
        prompt = context.get('system_prompt') or ''
        if prompt and prompt != self.system_prompt:
            self.system_prompt = prompt
            self.append({'type': 'system_prompt', 'content': prompt})

    def sync(self, context: dict):
        """把当前 context 完整刷到文件（系统提示词 + 新增消息）。"""
        self.sync_system_prompt(context)
        self.sync_messages(context)

    def sync_messages(self, context: dict):
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

    restored_messages, restored_prompt = load_session()
    context = {'system_prompt': restored_prompt or SYSTEM_PROMPT, 'messages': restored_messages}
    tools_list = builtin_tools()
    ui = Tui()
    log = SessionLog(SESSION_FILE)
    log.start(model, tools_list, context)   # 先把 meta / system prompt 落盘，再发任何请求
    flush_context = log.sync                # 每次调用模型前先落盘，保证 jsonl 不缺上下文

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

        # 斜杠指令：本地处理，不发给模型，也不会写进对话历史
        result = handle_command(first_line, context)
        if result is not None:
            if result: print(result)
            # /skill 会改动系统提示词，改动同样属于“发给模型的上下文”，必须落盘
            log.sync_system_prompt(context)
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
        log.sync_messages(context)   # 先把 user 消息落盘，避免中途崩掉时上下文缺尾巴

        signal = Signal()
        try:
            for ev in run_agent(model, context, tools_list, signal, before_request=flush_context):
                if ev['type'] == 'assistant_text': ui.print_text(ev['delta'])
                elif ev['type'] == 'reasoning_text': ui.print_reasoning(ev['delta'])
                elif ev['type'] == 'tool_call': ui.print_tool_call(ev['name'], ev['args'])
                elif ev['type'] == 'tool_result': ui.print_tool_result(ev['name'], ev['result'])
                elif ev['type'] == 'context_compacted':
                    # 压缩重写了 messages，立刻写一份完整快照（含摘要请求原文）
                    log.on_compaction(ev)
                    ui.print_notice(f"[context compacted] dropped {ev['dropped']} messages, kept {ev['kept']}")
                elif ev['type'] == 'turn_end':
                    if ev['stop_reason'] == 'max_tokens': ui.print_text('\n[output truncated by max_tokens]')
                    if ev['stop_reason'] == 'error': ui.print_text('\n[error occurred]')
                    ui.print_turn_end()
        except KeyboardInterrupt:
            signal.aborted = True
            print('\n[aborted by user]')

        # 本回合遗留的上下文（含中断 / 错误路径）再刷一遍，幂等
        log.sync(context)

if __name__ == '__main__':
    main()
