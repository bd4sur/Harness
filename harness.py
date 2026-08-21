from __future__ import annotations

# ==========================================
# src/llm.py
# 统一 LLM API —— 把 OpenAI Completions SSE 响应解析成四种事件流。只支持 OpenAI 兼容格式。
# ==========================================
import json
import codecs
import urllib.request
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

        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue

        if role == 'assistant':
            text = ''
            tool_calls = []
            for b in content:
                if b['type'] == 'text': text += b['text']
                elif b['type'] == 'tool_use':
                    tool_calls.append({
                        'id': b['id'], 'type': 'function',
                        'function': {'name': b['name'], 'arguments': json.dumps(b['input'], ensure_ascii=False)}
                    })
            c = text if text else (None if tool_calls else '')
            m = {'role': 'assistant', 'content': c}
            if tool_calls: m['tool_calls'] = tool_calls
            messages.append(m)
        else:
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
                chunk = response.read(2048)
                if not chunk: 
                    # 流结束，flush 解码器中可能残留的字节
                    final = decoder.decode(b'', final=True)
                    if final: buffer += final
                    break
                    
                # 增量解码：自动处理多字节字符边界，不完整的字节会缓存在 decoder 内部
                buffer += decoder.decode(chunk)
                
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
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
                        yield {'type': 'text_delta', 'delta': delta['content']}
                        
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

def build_assistant_message(text: str, tool_calls: list[dict]) -> dict:
    content = []
    if text: content.append({'type': 'text', 'text': text})
    for tc in tool_calls:
        content.append({'type': 'tool_use', 'id': tc['id'], 'name': tc['name'], 'input': tc['args']})
    return {'role': 'assistant', 'content': content}

def build_tool_result_message(results: list[dict]) -> dict:
    return {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': r['tool_use_id'], 'content': r['content']} for r in results]}


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

def compact_context(model: Model, context: dict, signal: Any = None):
    if (signal and signal.aborted) or len(context.get('messages', [])) < COMPACT_THRESHOLD:
        return

    old_msgs = context['messages'][:-KEEP_RECENT]
    recent_msgs = context['messages'][-KEEP_RECENT:]
    conv_text = '\n'.join([f"{m['role']}: {json.dumps(m['content'], ensure_ascii=False) if isinstance(m['content'], list) else m['content']}" for m in old_msgs])

    summary_ctx = {
        'system_prompt': '请将以下对话总结为简洁的上下文摘要，保留关键决策、已做的工作和待办事项。',
        'messages': [{'role': 'user', 'content': conv_text}]
    }

    summary, failed = '', False
    for ev in stream(model, summary_ctx, signal=signal):
        if ev['type'] == 'text_delta': summary += ev['delta']
        elif ev['type'] == 'error' or (ev['type'] == 'done' and ev['stop_reason'] == 'aborted'):
            failed = True; break

    if not failed and summary:
        context['messages'] = [{'role': 'user', 'content': f'[context summary]\n{summary}'}] + recent_msgs

def run_agent(model: Model, context: dict, tools: list[AgentTool], signal: Any = None) -> Generator[dict, None, None]:
    tool_map = {t.name: t for t in tools}
    tool_defs = [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools]

    while True:
        compact_context(model, context, signal)
        text, stop_reason, tool_calls = '', 'end_turn', []

        try:
            for ev in stream(model, context, tools=tool_defs, signal=signal):
                if ev['type'] == 'text_delta':
                    text += ev['delta']
                    yield {'type': 'assistant_text', 'delta': ev['delta']}
                elif ev['type'] == 'tool_call':
                    tool_calls.append({'id': ev['id'], 'name': ev['name'], 'args': ev['args']})
                    yield {'type': 'tool_call', 'id': ev['id'], 'name': ev['name'], 'args': ev['args']}
                elif ev['type'] == 'done':
                    stop_reason = ev['stop_reason']
                    if stop_reason == 'aborted':
                        context['messages'].append(build_assistant_message(text, []))
                        yield {'type': 'turn_end', 'stop_reason': 'aborted'}
                        return
                elif ev['type'] == 'error':
                    context['messages'].append(build_assistant_message(text, []))
                    yield {'type': 'assistant_text', 'delta': f"\n[error] {ev['error']}"}
                    yield {'type': 'turn_end', 'stop_reason': 'error'}
                    return
        except KeyboardInterrupt:
            if signal: signal.aborted = True
            context['messages'].append(build_assistant_message(text, []))
            yield {'type': 'turn_end', 'stop_reason': 'aborted'}
            return

        context['messages'].append(build_assistant_message(text, tool_calls))

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
import tempfile
from typing import Any

MAX_OUTPUT_LINES = 200
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
    try:
        res = subprocess.run(args['command'], shell=True, capture_output=True, text=True, timeout=30)
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
    def print_text(self, delta: str):
        sys.stdout.write(delta)
        sys.stdout.flush()

    def print_tool_call(self, name: str, args: dict):
        sys.stdout.write(f'\n[tool: {name}] {json.dumps(args, ensure_ascii=False)}\n')
        sys.stdout.flush()

    def print_tool_result(self, name: str, result: str):
        sys.stdout.write(f'[result: {name}] {result}\n')
        sys.stdout.flush()

    def print_turn_end(self):
        sys.stdout.write('\n')
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

SESSION_DIR = os.path.join(os.getcwd(), 'sessions')
SESSION_FILE = os.path.join(SESSION_DIR, f'session_{int(time.time() * 1000)}.jsonl')
SYSTEM_PROMPT = f"你是一个编码助手。用提供的工具读写文件和执行命令来完成任务。先阅读再修改，修改后可运行命令验证。你的工作目录是：{os.getcwd()}/\n"

class Signal:
    def __init__(self): self.aborted = False

def load_session() -> list[dict]:
    if not os.path.exists(SESSION_FILE): return []
    messages = []
    with open(SESSION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: messages.append(json.loads(line))
                except: pass
    return messages

def main():
    api_key = os.environ.get('NANOPI_API_KEY')
    if not api_key:
        print('未设置 NANOPI_API_KEY 环境变量，默认为空（通过 export NANOPI_API_KEY=xxx 设置）', file=sys.stderr)
        api_key = "sk-dummykey"

    model = Model(
        api_key=api_key,
        model=os.environ.get('NANOPI_MODEL', 'deepseek-v4-flash-0731'),
        base_url=os.environ.get('NANOPI_BASE_URL', 'http://192.168.10.50:8888/v1'),
        max_tokens=4096
    )

    context = {'system_prompt': SYSTEM_PROMPT, 'messages': load_session()}
    persisted_count = len(context['messages'])
    tools_list = builtin_tools()
    ui = Tui()

    print("Agent 已启动 (输入 Ctrl+C 可随时中断当前任务，连续两次 Ctrl+C 退出程序)")
    while True:
        try:
            text = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break

        if not text: continue

        # 斜杠指令：本地处理，不发给模型，也不会写进对话历史
        result = handle_command(text, context)
        if result is not None:
            if result: print(result)
            continue

        context['messages'].append({'role': 'user', 'content': text})

        signal = Signal()
        try:
            for ev in run_agent(model, context, tools_list, signal):
                if ev['type'] == 'assistant_text': ui.print_text(ev['delta'])
                elif ev['type'] == 'tool_call': ui.print_tool_call(ev['name'], ev['args'])
                elif ev['type'] == 'tool_result': ui.print_tool_result(ev['name'], ev['result'])
                elif ev['type'] == 'turn_end':
                    if ev['stop_reason'] == 'max_tokens': ui.print_text('\n[output truncated by max_tokens]')
                    if ev['stop_reason'] == 'error': ui.print_text('\n[error occurred]')
                    ui.print_turn_end()
        except KeyboardInterrupt:
            signal.aborted = True
            print('\n[aborted by user]')

        os.makedirs(SESSION_DIR, exist_ok=True)
        with open(SESSION_FILE, 'a', encoding='utf-8') as f:
            for msg in context['messages'][persisted_count:]:
                f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        persisted_count = len(context['messages'])

if __name__ == '__main__':
    main()

