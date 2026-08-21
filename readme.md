## 项目概述

知识库 RAG 系统，旨在为垂直领域（如电子产品手册、维修指南、技术文档等）提供精准、智能的知识检索与问答服务。
支持文档导入（PDF→Markdown→向量化→入库）和智能检索问答（多路召回→RRF 融合→Rerank→LLM 生成）。

## 常用命令

```bash
# 激活虚拟环境
cd knowledge && source .venv/Scripts/activate

# 启动基础设施（Milvus/Etcd/MinIO/MongoDB/Attu）
docker-compose -f knowledge/docker-compose.yml up -d

# 启动导入服务（端口 8000）
python -m knowledge.api.import_router

# 启动查询服务（端口 8001）
python -m knowledge.api.query_router
```

- 安装依赖：`pip install -r knowledge/requirements.txt`（Python 3.12）
- 没有测试框架、lint 配置或 CI pipeline
- 测试脚本在 `knowledge/test/`，均为手动运行的编号脚本
- Attu（Milvus 管理界面）运行在 `http://localhost:7000`

## 架构

```
api/          FastAPI 路由、CORS、静态文件挂载
core/         DI 容器（functools.cache 单例）、文件路径
service/      业务用例层，编排 processor 和 utils
processor/    LangGraph 有状态工作流（核心逻辑）
  import_process/   文档导入管线
  query_process/    检索问答管线
schema/       Pydantic 请求/响应模型
prompt/       LLM prompt 模板
utils/        跨模块工具（Milvus/MinIO/MongoDB/Embedding 客户端、SSE 推送）
front/        静态 HTML 页面（chat.html, import.html）
```

**两个独立 FastAPI 应用**，各自启动在独立端口和进程：
- **导入服务** (`api/import_router.py`, 端口 8000)：文件上传 → 后台 LangGraph → 轮询状态
- **查询服务** (`api/query_router.py`, 端口 8001)：POST `/query` 支持流式(SSE)和非流式两种模式

### 导入管线 (Import Graph)

```
entry → [PDF? pdf_to_md] → md_img → document_split → item_name_recognition
→ bge_embedding → import_milvus → END
```

- `entry` 节点根据文件类型条件路由（PDF 走 MinerU 转换，MD 直接跳转）
- `md_img` 对文档中的图片调用 VLM 生成描述文本
- `item_name_recognition` 用 LLM 从 chunk 中提取商品名称
- `bge_embedding` 使用 BGE-M3 模型生成 dense + sparse 双路向量
- `import_milvus` 将向量存入 Milvus Collection

### 查询管线 (Query Graph)

```
item_name_confirm ──[有答案]──> answer_output
       │
       └──[无答案]──> multi_search ──┬── vector_search (BGE-M3 混合检索)
                                     ├── hyde_search (HyDE 假设文档检索)
                                     └── web_search_mcp (DashScope 联网搜索)
                                     └──> join → rrf → rerank → answer_output
```

- `item_name_confirm` 先确认商品名称，若问题无需检索直接回答则跳过搜索
- 三路检索并行执行，通过虚拟节点 `multi_search`/`join` 实现分发与汇合
- `rrf` 对多路结果做 RRF 融合排序
- `rerank` 使用 FlagReranker 精排
- `answer_output` 拼接上下文调用 LLM 生成答案，支持 SSE 流式输出

## 核心设计模式

### 客户端单例 (`utils/client/`)

`BaseClientManager` 实现双重检查锁定（double-checked locking）的线程安全懒加载。`AIClients` 和 `StorageClients` 继承它，每个外部依赖（OpenAI/BGE/FlagReranker/MinIO/Milvus/MongoDB）作为类属性持有独立锁。通过 `AIClients()` / `StorageClients()` 获取实例。

### BaseNode 模板方法 (`processor/*/base.py`)

所有 LangGraph 节点继承 `BaseNode(ABC)`，子类覆写 `process(state)`。基类 `__call__` 注入统一日志、任务状态追踪、节点耗时、SSE 进度推送（仅查询管线）、异常包装。

### 配置管理 (`processor/*/config.py`)

`@dataclass` 配置类，字段使用 `field(default_factory=lambda: os.getenv(...))` 做运行时懒加载。模块级 `get_config()` 工厂函数持有全局单例（导入配置的锁被注释掉，非线程安全）。

### SSE 流式推送 (`utils/sse_util.py`)

内存中的 `task_id → queue.Queue` 映射。生产者 `push_sse_event()` 往队列写事件（progress/delta/final），消费者 `sse_generator()` 从 `/stream/{task_id}` 端点读取并 yield SSE 事件。

## 注意事项

- **代码和注释使用中文**编写
- `.env` 已 gitignored，但实际配置指向远程主机 `192.168.200.3`；`docker-compose.yml` 定义的是本地容器，两者不一致
- **嵌入模型实际使用本地 BGE-M3**（`utils/embedding_util.py` 和 `utils/client/ai_clients.py`），`.env` 中的 `EMBEDDING_DIM=1536` 和 `EMBEDDING_MODEL=text-embedding-v4` 并未被代码实际使用
- `query_process/config.py` 的 `default_model` 读取环境变量 `MODEL`，但 `.env` 中键名为 `LLM_DEFAULT_MODEL`——该字段会读到空字符串
- `processor/import_process/nodes/ducment_split.py` 文件名有拼写错误（应为 `document_split`），不要修改除非修复连锁引用
