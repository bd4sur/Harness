# Harness 项目说明

## 项目简介

Harness 是一个轻量级的**命令行 Agent 框架**，把 OpenAI 兼容的大语言模型（LLM）与本地文件/命令工具连接起来，形成一个可以在终端里执行编码任务的交互式助手。整个项目以单个 Python 文件 `harness.py` 为核心，没有复杂依赖，适合作为个人本地 Agent 的最小可运行原型。

核心能力：

- **流式对话**：实时显示模型返回的文本与推理过程。
- **工具调用**：支持读文件、写文件、精确编辑文件、执行 Shell 命令。
- **Agent 循环**：模型可以在一次任务中反复调用工具，直到自行判断完成。
- **会话持久化**：对话按运行时间写入 `sessions/session_<时间戳>.jsonl`。
- **上下文压缩**：当对话轮数超过阈值，自动用模型对历史进行摘要，避免 token 无限增长。
- **斜杠指令与 Skills**：用户可以输入 `/` 开头的本地命令，也可以动态加载 `skills/` 目录下的 Markdown 文件到系统提示词。

## 文件结构

```
/mnt/d/Desktop/repos/Harness/
├── harness.py          # 整个项目的唯一入口，包含 LLM、Agent、工具、命令、CLI 五部分
├── skills/
│   └── grilling.md     # 示例 skill：通过“拷问式”提问帮用户完善计划
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
| `read_file` | 读取文件内容，大文件只返回最后 200 行 | `path` |
| `write_file` | 写入/覆盖文件，必要时自动创建父目录 | `path`, `content` |
| `edit` | 精确替换一段文本，`old_string` 必须唯一匹配 | `path`, `old_string`, `new_string` |
| `run_bash` | 执行 Shell 命令，执行前需要用户在终端确认；输出保留最后 200 行 | `command` |

> **安全提示**：`run_bash` 会弹出现场确认，避免模型在无人干预时直接运行危险命令。其它文件工具则直接执行，因此建议在受信任的工作目录中使用。

## 斜杠指令与 Skills

以 `/` 开头的输入会由本地解析执行，**不会发送给 LLM**，也不会写入会话历史。

| 指令 | 作用 |
|---|---|
| `/help` | 显示可用指令 |
| `/skills` | 列出 `skills/` 目录下可用的 skill |
| `/skill <name>` | 显式加载 `skills/<name>.md` 并追加到 system prompt |
| `/<name>` | 未注册的内置指令默认当作 skill 名，等价于 `/skill <name>` |

Skill 文件使用 Markdown 格式，顶部可用 `---` 包裹 YAML frontmatter（如 `name`、`description`），加载时 frontmatter 会被自动剔除，正文追加进系统提示词。

仓库自带的 `grilling` skill 会让 Agent 进入“设计树拷问”模式：通过多轮问答帮用户把想法和决策逐层展开，直到没有未解决的隐含假设。

## 会话与上下文

- 每次启动时，Harness 会加载最近生成的 `sessions/session_*.jsonl` 文件（如果存在），作为初始历史。
- 新的对话会增量追加到同一个会话文件中。
- 当历史消息数超过 `COMPACT_THRESHOLD`（默认 50 条）时，Agent 会调用一次 LLM 对早期对话做摘要，仅保留最近的 `KEEP_RECENT`（默认 20 条）原始消息，从而控制上下文长度。
- 摘要时如果保留的最近消息以 `tool` 结果开头，`compact_context` 会向前扩展，把产生该 tool 的 `assistant` 消息一并保留，避免 OpenAI 兼容 API 因 tool 消息前缺少对应 `tool_calls` 而报 400。

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
   - 定义 `AgentTool` 数据结构并实现 4 个内置工具。
   - 命令输出超过 200 行时，会写入临时文件并只返回末尾内容。

4. **命令层（`src/commands.py`）**
   - 解析以 `/` 开头的输入。
   - 维护内置指令注册表，未知指令回退为 skill 加载。

5. **CLI/TUI 层（`src/cli.py` + `src/tui.py`）**
   - `Tui` 负责把流式事件渲染到终端，包括普通文本、推理文本（显示为 `[思考: ...]`）、工具调用与结果。
   - `cli.py` 是程序入口，负责配置模型、加载会话、启动交互循环。

## 扩展与开发建议

- **增加工具**：在 `builtin_tools()` 中新增 `AgentTool(name, description, parameters, execute)`，并在工具描述里说明参数。
- **增加指令**：在 `builtin_commands()` 注册新的 handler，接收 `(args, context)` 并返回展示文本。
- **增加 Skill**：在 `skills/` 目录新建 `<name>.md`，内容会作为额外系统提示追加。
- **修改提示词**：调整 `cli.py` 中的 `SYSTEM_PROMPT`。
- **调试 API 问题**：可以先通过 `curl` 直接测试 `HARNESS_BASE_URL` 是否支持 OpenAI 兼容的 `/chat/completions` 流式接口。

## 注意事项

- 本项目仅支持 OpenAI 兼容的流式 Chat Completions 接口。
- 模型API 端点需要正确处理 SSE 流、`finish_reason` 以及 `tool_calls` 增量字段。
- 由于 `run_bash` 需要终端交互确认，在完全非交互环境运行可能会卡住。


