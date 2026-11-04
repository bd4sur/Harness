# Harness 项目说明

## 项目简介

Harness 是一个轻量级的**命令行 Agent 框架**，把 OpenAI 兼容的大语言模型（LLM）与本地文件/命令工具连接起来，形成一个可以在终端里执行编码任务的交互式助手。整个项目以单个 Python 文件 `harness.py` 为核心，没有复杂依赖，适合作为个人本地 Agent 的最小可运行原型。

核心能力：

- **流式对话**：实时显示模型返回的文本与推理过程。
- **工具调用**：支持读文件、写文件、精确编辑文件、执行 Shell 命令。
- **Agent 循环**：模型可以在一次任务中反复调用工具，直到自行判断完成。
- **会话持久化**：对话按运行时间写入 `sessions/session_<时间戳>.jsonl`，包括系统提示词与请求的固定部分（模型名、`max_tokens`、工具 schema），最终发给模型的上下文可从该文件完整还原。
- **上下文压缩**：当对话轮数超过阈值，自动用模型对历史进行摘要，避免 token 无限增长。

## 文件结构

```
/mnt/d/Desktop/repos/Harness/
├── harness.py          # 整个项目的唯一入口，包含 LLM、Agent、工具、TUI、CLI 五部分
├── sessions/           # 运行时自动创建，保存对话历史（JSON Lines）
└── AGENTS.md           # 本说明文档
```

## 运行方式

在仓库根目录下执行：

```bash
python harness.py
```

启动后会出现 `>` 提示符，输入问题或任务即可。多行输入时，后续行以 `+ ` 提示，输入一个**空行**表示结束。

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `HARNESS_API_KEY` | LLM API 的密钥 | `sk-0` |
| `HARNESS_MODEL` | 模型名称 | `deepseek-v4-flash-0731` |
| `HARNESS_BASE_URL` | OpenAI 兼容 API 的基础 URL | `http://192.168.10.50:8888/v1` |
| `HARNESS_MAX_TOKENS` | 每次请求的最大生成 token 数 | `1048576` |
| `HARNESS_SESSION` | 会话文件路径。不设则续接 `sessions/` 里最新的一份（不存在则新建）；设为 `none` 强制新开会话；设为某个路径则读写该文件 | 续接最新 |

示例：

```bash
export HARNESS_API_KEY=your_api_key
export HARNESS_BASE_URL=https://api.example.com/v1
python harness.py
```

## 内置工具

Harness 只提供 4 个最基础的工具，覆盖“读、写、改、验证”的最小闭环。

| 工具名 | 功能 | 关键参数 |
|---|---|---|
| `read_file` | 读取文件内容，超长文件只返回末尾 5000 行 | `path` |
| `write_file` | 写入/覆盖文件，必要时自动创建父目录 | `path`, `content` |
| `edit` | 精确替换一段文本，`old_string` 必须唯一匹配 | `path`, `old_string`, `new_string` |
| `run_bash` | 执行 Shell 命令，执行前需要用户在终端确认；输出保留最后 5000 行 | `command` |

> **安全提示**：`run_bash` 会弹出现场确认，避免模型在无人干预时直接运行危险命令。其它文件工具则直接执行，因此建议在受信任的工作目录中使用。

## 会话与上下文

- 每次启动时，Harness 会加载最近生成的 `sessions/session_*.jsonl` 文件（如果存在），作为初始历史；`HARNESS_SESSION` 可指定别的路径或强制新开会话。
- 新的对话会增量追加到同一个会话文件中；续接时 `SessionLog` 的增量游标从已恢复的消息数起算，历史不会被重复写第二遍。
- 续接不会破坏前缀缓存：系统提示词是常量，`reasoning_content` 也随 assistant 消息一起存盘，因此恢复出来的历史与上次实际发出的请求逐 token 相同。
- 如果上次是被强杀/崩溃结束的，文件可能停在「`assistant` 已发起 `tool_calls`、工具结果还没落盘」的位置，`load_session` 会把这段残缺尾巴丢掉，避免 OpenAI 兼容 API 报 400。
- 当历史消息数超过 `COMPACT_THRESHOLD`（默认 50 条）时，Agent 会调用一次 LLM 对早期对话做摘要，仅保留最近的 `KEEP_RECENT`（默认 20 条）原始消息，从而控制上下文长度。
- 摘要时如果保留的最近消息以 `tool` 结果开头，`compact_context` 会向前扩展，把产生该 tool 的 `assistant` 消息一并保留，避免 OpenAI 兼容 API 因 tool 消息前缺少对应 `tool_calls` 而报 400。

### session jsonl 记录格式

**原则：凡是发给模型的上下文（system prompt、messages、tools、max_tokens、model）都必须能在 session 文件里找到，按行读回即为真实上下文顺序。** 文件每行一个 JSON 对象，共五种记录：

| 记录 | 形态 | 含义 |
|---|---|---|
| 消息 | `{"role": ..., "content": ...}` | 与 `context['messages']` 中的一条消息一一对应（assistant 的 `tool_calls`、tool 的 `tool_result` 均在 `content` 旁边原样保存） |
| `meta` | `{"type": "meta", "model", "base_url", "max_tokens", "stream", "tools", "started_at"}` | 请求的固定部分：启动时写一次，含全部工具 schema |
| `system_prompt` | `{"type": "system_prompt", "content": ...}` | 系统提示词**全文**；启动时写一次 |
| `snapshot` | `{"type": "snapshot", "reason", "messages", ...}` | 上下文被重写（压缩）后的**完整**消息列表；压缩时还带 `dropped`、`kept` 与 `summary_request`（压缩时额外发给模型的摘要请求原文） |
| `usage` | `{"type": "usage", "prompt_tokens", "cached_tokens", "completion_tokens"}` | 每次请求的 token 用量，`cached_tokens` 是命中前缀缓存的 prompt 部分 |

写入时机保证「先落盘、后请求」：

- 启动时先写 `meta` + `system_prompt`，再发任何请求。
- `run_agent` 每次调用模型前都会通过 `before_request` 回调把当前 context 刷入文件，因此会话中途的每一次请求都能在文件里对上号。
- 用户消息落盘后再请求；压缩（上下文重写）时立刻写 `snapshot` 并重置增量游标。
- 读回（`load_session`）时：消息记录按序追加，`snapshot` 记录整份替换 `messages`，最后一条 `system_prompt` 作为当前系统提示词；`meta` 只用于审计。压缩后游标重置，避免了旧实现（`persisted_count` 只增不减）在压缩后不再写盘的问题。
- 以 `role` 字段区分消息记录与其它记录，因此旧版本（只有纯消息行）的 session 文件仍可正常读取。

## 主要模块说明

`harness.py` 虽然是一个文件，但内部按职责划分为五个逻辑模块：

1. **LLM 层（`src/llm.py`）**
   - 把本地 `context` 格式转换为 OpenAI 兼容的 `messages` 格式。
     - 对 assistant 消息做规范化：无内容时 `content` 退化为空字符串 `""`；包含 `tool_calls` 时 `content` 会被清空为 `""`，避免部分 OpenAI 兼容 API 报 400。
   - 通过 SSE 流式解析，将模型返回拆成 `text_delta`、`reasoning_delta`、`tool_call`、`done`、`error` 等事件。
   - 支持工具调用和提前中止（`signal`）。

2. **Agent 层（`src/agent.py`）**
   - 核心就是一个 `while True` 循环：调用 LLM → 解析工具调用 → 执行工具 → 把结果喂回模型。
   - 处理 `max_tokens` 截断、用户 `Ctrl+C` 中断、错误等边界情况。
   - 触发上下文压缩。

3. **工具层（`src/tools.py`）**
   - `AgentTool` 数据结构 + 一张 `TOOL_SPECS` 表（`name, description, parameters, execute` 四元组）驱动 4 个内置工具；`builtin_tools()` 只是把表拼成 `AgentTool` 列表。新增工具只改表。
   - 命令输出超过 5000 行时，会写入临时文件并只返回末尾内容；临时文件按**内容哈希**命名（不是 pid/序号），保证同样的输出得到同样的工具结果字符串，跨会话也能复用前缀缓存。

4. **CLI/TUI 层（`src/cli.py` + `src/tui.py`）**
   - `Tui` 负责把流式事件渲染到终端。所有“一整行”的输出（`text` / `result` / `turn_end` / `notice` / `tool_call`）都走同一个 `print(kind, *args)`，具体格式在那张 `LINE` 表里；只有推理文本（`print_reasoning`，带 `[Thinking: ...]` 状态机）和 `print_usage` 因为有自己的状态/计算而单独存在。
   - `cli.py` 是程序入口，负责配置模型、加载会话、启动交互循环。

## 扩展与开发建议

- **增加工具**：在 `TOOL_SPECS` 表里加一条 `(name, description, parameters, execute)`，并在 description 里说明参数；不需要改 `builtin_tools()` 或 agent 循环。
- **修改提示词**：调整 `cli.py` 中的 `SYSTEM_PROMPT`。它必须保持为**常量**：不能写入时间、`cwd`、主机名等每次启动都会变的内容，否则服务端前缀缓存从 system 开头就失效；需要环境信息时，让模型自己用 `run_bash` 去查（提示词里已经列出了对应的命令）。
- **改前缀缓存策略**：消息只能**追加**，不能修改已发出的历史（唯一合法例外是压缩）；压缩（`snapshot`）会把缓存从头废掉，所以 `COMPACT_THRESHOLD` 不要设得太小。命中率看每回合的 `[cache N%]` 输出或 session 里的 `usage` 记录。
- **调试 API 问题**：可以先通过 `curl` 直接测试 `HARNESS_BASE_URL` 是否支持 OpenAI 兼容的 `/chat/completions` 流式接口。

## 注意事项

- 本项目仅支持 OpenAI 兼容的流式 Chat Completions 接口。
- 模型API 端点需要正确处理 SSE 流、`finish_reason` 以及 `tool_calls` 增量字段。
- 由于 `run_bash` 需要终端交互确认，在完全非交互环境运行可能会卡住。

## 已知坑（代码里只留一行指针，详情在这里）

这四条是“删了就一定出问题”的硬约束，每一条都是对具体 API 行为踩过坑之后的产物：

1. **assistant 消息的 `content` 与空消息**
   - 带 `tool_calls` 时 `content` 必须是空字符串 `""`（`context_to_openai_messages`）。
   - 既没有 `text` 也没有 `tool_calls` 的 assistant 消息**不要写入历史**（`build_assistant_message` 返回 `None`）。Kimi（`api.kimi.com`）会把 `{"role":"assistant","content":""}` 判成非法格式直接 400；详见文末「遗留问题」。
   - 空转轮次（模型什么都没输出）不 append 任何消息。

2. **tool 消息前面必须有产生它的 assistant**
   - OpenAI 兼容 API 要求每个 `role=tool` 紧跟带对应 `tool_calls` 的 assistant。
   - 压缩时如果“保留的最近 KEEP_RECENT 条”以 `tool` 开头，`compact_context` 必须向前扩展，把发起它的 assistant 一起保下来，否则下一次请求 400。
   - 中断（abort）时也要给未执行的 tool_call 补一条 `error: aborted` 结果，保证 `tool_calls` 与 tool 结果一一对应。

3. **续接会话时的残缺尾巴**
   - 落盘顺序是 user →（请求前 sync）→ assistant(tool_calls) → 执行工具 → tool 结果。若在工具执行期间被 kill，文件就停在“assistant 有 tool_calls 但没有 tool 结果”的位置，`load_session` 必须把这段尾巴整段丢掉（连同发起它的 assistant）。
   - 续接后 `SessionLog` 的增量游标必须从已恢复的消息数起算，否则历史会被重复追加一遍。

4. **前缀缓存的三条前提**（命中率看 `[cache N%]`）
   - 系统提示词必须是常量，不能含时间、`cwd`、主机名；要环境信息就让模型自己 `run_bash` 去查。
   - 历史只能追加；`reasoning_content` **全部回传**、无开关，但只拼在历史末尾，不改已有前缀。
   - 工具结果字符串必须确定性：截断输出按**内容哈希**命名临时文件，而不是 pid/序号，否则同一段输出每次都是新字符串，跨会话缓存永远无法复用。

5. **`tool_calls` 是流式增量拼接的**
   - `stream` 把同一个 `index` 的 `function.arguments` 片段累加后再 `json.loads`；中断时不能把缓冲区里读了一半的 args 当成完整调用抛给 agent（会在工具里执行半截命令）。












# 遗留问题

这个请求体报 400，核心原因不是 `content: []`（那个已经修掉了），而是 **存在两条"空壳" assistant 消息**：

```json
{"role": "assistant", "content": ""}
```

请求体里出现了两次：
1. 在 `tool_sG0z1hak...` 的 tool result 之后
2. 在 user 说 `"你好"` 之后

---

### 为什么 `{"role": "assistant", "content": ""}` 会 400？

OpenAI 兼容规范里，`assistant` 消息的 `content` 允许 `string` 或 `null`。但 **Kimi（`api.kimi.com`）的校验比 OpenAI 更严格**：

> 当 `assistant` 消息**没有 `tool_calls`** 时，`content` 不能是空字符串 `""`。  
> 空字符串会被服务端视为"这条消息没有任何有效信息"，直接判为格式错误。

换句话说：
- `{"role": "assistant", "content": "", "tool_calls": [...]}` → **合法**（发起工具调用时 content 为空是正常的）
- `{"role": "assistant", "content": ""}` → **某些实现里不合法**（没有 tool_calls 又没有内容，等于白占一条消息）

你的 harness 在 `build_assistant_message(text, [])` 里用了 `text or ''`，导致空文本被硬编码成 `""`。当模型某一轮什么也没输出（比如用户说 `"你好"`，模型只回了空），这条空消息被写进 session，下次请求带上它就炸了。

---

### 另一个助推因素：`max_tokens` 过大

```json
"max_tokens": 1048576
```

Kimi API 的上下文窗口通常只有 128K 或 256K。`max_tokens` 设成 1M 会让 `prompt_tokens + max_tokens` 的总和远超模型上限，API 也可能因此拒掉请求。错误信息通常是 `max_tokens exceeds limit` 或类似的 `invalid_request_error`。

---

### 修复建议

**1. 空 assistant 消息的处理**

把 `build_assistant_message` 里对空文本的处理改掉：

```python
# 原来
return {'role': 'assistant', 'content': text or '', 'tool_calls': tool_calls or []}

# 改为：没有 tool_calls 且没有 text 时，content 用 null 或干脆不存这条消息
if not text and not tool_calls:
    # 方案 A：content 设为 null（Kimi 通常接受 null）
    return {'role': 'assistant', 'content': None}
    # 方案 B：更保险——这种空消息不写入历史，直接跳过
```

**推荐方案 B**：如果一轮对话模型既没输出文本也没调用工具，说明这一轮是"空转"，不应该往 `context['messages']` 里 append 任何东西。

**2. `max_tokens` 设个合理的上限**

```python
max_tokens=min(int(os.environ.get('HARNESS_MAX_TOKENS', '8192')), 8192)
```

或者至少不能超过常见模型的上下文窗口余量。

---

### 总结

| 问题 | 位置 | 影响 |
|------|------|------|
| `content: ""` 且无 `tool_calls` | 两条 assistant 消息 | 被 Kimi API 视为无效消息，400 |
| `max_tokens: 1048576` | 请求体顶层 | 可能触发上限校验，400 |

优先修第一条，因为那是格式层面的硬伤；第二条是参数层面的问题，修完更稳健。