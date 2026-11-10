from __future__ import annotations
import readline

# ==========================================
# src/llm.py
# 统一 LLM API —— 把 OpenAI Completions SSE 响应解析成四种事件流。只支持 OpenAI 兼容格式。
# ==========================================
import json, urllib.request, urllib.error
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
            # thinking 模型需要把上一轮的思考过程原样带回去
            if msg.get('reasoning_content'):
                out['reasoning_content'] = msg['reasoning_content']
            messages.append(out)
        else:  # role == 'tool'：content 是 tool_result 块列表
            for b in content:
                messages.append({'role': 'tool', 'tool_call_id': b['tool_use_id'], 'content': b['content']})
    return messages

def _chars(kind: str, text: str) -> Generator[dict, None, None]:
    """把一段文本拆成逐字符增量事件，让刷新粒度不依赖服务端 delta 的切分方式。"""
    for ch in text: yield {'type': kind, 'delta': ch}

def stream(model: Model, context: dict, tools: Optional[list[dict]] = None, signal: Any = None) -> Generator[dict, None, None]:
    url = f"{model.base_url}/chat/completions"
    body: dict[str, Any] = {'model': model.model, 'stream': True, 'messages': context_to_openai_messages(context)}
    if model.max_tokens: body['max_tokens'] = model.max_tokens
    if tools: body['tools'] = [{'type': 'function', 'function': t} for t in tools]

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {model.api_key}'}
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode('utf-8'), headers=headers, method='POST')

    tool_call_buffers: dict[int, dict[str, str]] = {}
    stop_reason = 'end_turn'
    try:
        with urllib.request.urlopen(req) as response:
            # SSE 是一行一个事件，response 本身就是行迭代器：阻塞读一行、解一行、yield 一个事件。
            # 不需要增量解码器 + 缓冲区状态机（多字节字符不含 0x0A，按行切永远是完整字符）。
            for raw in response:
                # 用户中断：直接收摊，不再把只收到一半的 tool_calls 当完整调用抛出去
                if signal and signal.aborted:
                    yield {'type': 'done', 'stop_reason': 'aborted'}
                    return

                line = raw.decode('utf-8', 'ignore').strip()
                if not line.startswith('data: '): continue
                data = line[6:]
                if data == '[DONE]': break

                try: chunk_json = json.loads(data)
                except json.JSONDecodeError: continue

                choice = (chunk_json.get('choices') or [{}])[0]
                if not choice: continue

                delta = choice.get('delta', {})
                if delta.get('content'):
                    yield from _chars('text_delta', delta['content'])
                reasoning = delta.get('reasoning_content') or delta.get('reasoning')
                if reasoning:
                    yield from _chars('reasoning_delta', reasoning)

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

def build_tool_result_message(results: list[dict]) -> dict:
    return {'role': 'tool', 'content': [{'type': 'tool_result', 'tool_use_id': r['tool_use_id'], 'content': r['content']} for r in results]}


# ==========================================
# src/agent.py
# Agent Loop —— 整个项目的灵魂：一个 while 循环。
# ==========================================
import json
from types import SimpleNamespace
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
        return None

    messages = context['messages']
    cut = len(messages) - KEEP_RECENT
    if messages[cut]['role'] == 'tool':
        # 保留段不能以 tool 开头：缺了发起它的 assistant，API 会报 400。退到最近的一条 assistant。
        cut = next((i for i in range(cut - 1, -1, -1) if messages[i]['role'] == 'assistant'), cut)
        # 极端情况：整段历史里找不到发起者，就丢掉开头的孤立 tool 消息
        while cut < len(messages) and messages[cut]['role'] == 'tool': cut += 1
    old_msgs, recent_msgs = messages[:cut], messages[cut:]

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
        return {'dropped': len(old_msgs), 'kept': len(recent_msgs)}
    return None

def _append_assistant(context: dict, text: str, tool_calls: list[dict], reasoning: str = ''):
    """构建 assistant 消息并追加进 context；既无文本也无 tool_calls 的空转轮次不写历史（见 AGENTS.md）。"""
    if not text and not tool_calls:
        return
    msg = {'role': 'assistant', 'content': text or ''}
    if tool_calls:
        msg['tool_calls'] = tool_calls
    # DeepSeek / Kimi 的 thinking 模型要求把上一轮的 reasoning_content 原样放回 assistant 消息，
    # 否则报 400 或推理链断裂。
    if reasoning:
        msg['reasoning_content'] = reasoning
    context['messages'].append(msg)

def run_agent(model: Model, context: dict, tools: list[AgentTool], signal: Any = None) -> Generator[dict, None, None]:
    tool_map = {t.name: t for t in tools}
    tool_defs = [{'name': t.name, 'description': t.description, 'parameters': t.parameters} for t in tools]

    while True:
        compact_info = compact_context(model, context, signal)
        if compact_info:
            yield {'type': 'context_compacted', **compact_info}
        text, reasoning, stop_reason, tool_calls = '', '', 'end_turn', []

        for ev in stream(model, context, tools=tool_defs, signal=signal):
            if ev['type'] == 'text_delta':
                text += ev['delta']
                yield {'type': 'assistant_text', 'delta': ev['delta']}
            elif ev['type'] == 'reasoning_delta':
                reasoning += ev['delta']            # 累起来，用于回传给 thinking 模型
                yield {'type': 'reasoning_text', 'delta': ev['delta']}
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

        # 工具结果统一在这里构造：max_tokens 截断、正常执行、已中断三种情况只差 content 从哪来，
        # 收集完 results 后只 append 一次、只上报一遍。
        truncated = stop_reason == 'max_tokens'
        results, stop = [], False
        for tc in tool_calls:
            if truncated:
                res = f'error: output truncated by max_tokens, tool "{tc["name"]}" args may be incomplete.'
            elif stop:
                res = 'error: aborted'
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
            if signal and signal.aborted: stop = True

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
    # 用内容哈希命名，而不是 pid/计数器：同样的输出得到同样的工具结果字符串，行为可复现。
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


# ==========================================
# src/cli.py
# 拼装层 —— 把 llm / agent / tui / tools 粘起来，是唯一入口。
# ==========================================

# 注意：系统提示词必须是常量，不能含时间、路径、主机名等每次启动都会变的内容。
# 需要环境信息时让模型自己调 run_bash 去查（见下方“环境信息”）。
SYSTEM_PROMPT = """你是一个严谨的本地编码助手：通过工具与用户协作，高质量完成代码相关任务。

环境信息：你不掌握任何运行环境的先验事实（时间、工作目录、系统版本等），需要时一律用 run_bash 现查（date、pwd、uname -a、python3 -V、echo $HOME），不要拍脑袋假设。本程序的模型配置固化在 main() 的常量里，不读环境变量。

核心工作流（必须遵守）：
1. 先读：改任何文件前，先用 read_file 读目标文件与相关上下文。
2. 再计划：说明要做什么、分几步、每步改什么。
3. 最小修改：优先 edit 精确替换；只有新建文件或必须整体重写才用 write_file。
4. 后验证：改完必须跑相关命令（测试、编译、lint、运行脚本）验证结果。
5. 汇报：用一句话总结本回合做了什么、验证结果如何。

错误处理：edit 报 old_string 不匹配就先重新 read_file 核对再重试；run_bash 失败先读 stderr，判断是命令问题、环境问题还是预期内错误；不要用同样的参数连续重试同一工具。

输出风格：回复简洁、结构化；思考过程用 [Thinking: ...] 展示，对外输出聚焦在行动和结果上。

安全边界：不主动删除用户文件，不执行可能破坏工作目录的命令（rm -rf、重定向覆盖关键文件等）；需要覆盖或删除时先征得用户同意。
"""

def Signal() -> SimpleNamespace:
    """只带一个 aborted 标志；Ctrl+C 置位后各层统一收摊。"""
    return SimpleNamespace(aborted=False)

def main():
    # 配置全部固化在这里，不读任何环境变量。
    model = Model(
        api_key='sk-0',
        model='deepseek-v4-flash-0731',
        base_url='http://192.168.10.50:8888/v1',
        max_tokens=65536,
    )

    context = {'system_prompt': SYSTEM_PROMPT, 'messages': []}
    tools_list = builtin_tools()
    ui = Tui()

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

        signal = Signal()
        try:
            for ev in run_agent(model, context, tools_list, signal):
                if ev['type'] == 'assistant_text': ui.print('text', ev['delta'])
                elif ev['type'] == 'reasoning_text': ui.print_reasoning(ev['delta'])
                elif ev['type'] == 'tool_call': ui.print('tool_call', ev['name'], ev['args'])
                elif ev['type'] == 'tool_result': ui.print('result', ev['name'])
                elif ev['type'] == 'context_compacted':
                    ui.print('notice', f"[context compacted] dropped {ev['dropped']} messages, kept {ev['kept']}")
                elif ev['type'] == 'turn_end':
                    if ev['stop_reason'] == 'max_tokens': ui.print('text', '\n[output truncated by max_tokens]')
                    if ev['stop_reason'] == 'error': ui.print('text', '\n[error occurred]')
                    ui.print('turn_end')
        except KeyboardInterrupt:
            signal.aborted = True
            print('\n[aborted by user]')

if __name__ == '__main__':
    main()
