# RAG 知识库问答系统

## 项目简介

本项目是一个基于 RAG 的知识库问答系统，面向论文、报告和企业文档场景，支持 PDF / MD 文档解析、对象存储、向量检索、图谱存储和问答生成，实现从非结构化文档到知识库问答的端到端流程。

## 技术栈

- 后端框架：FastAPI, Python
- RAG 编排：LangChain, LangGraph
- 文档解析：MinerU
- 向量数据库：Milvus
- 图数据库：Neo4j
- 文档与记录存储：MongoDB
- 对象存储：MinIO
- 工程化：Docker, OmegaConf

## 核心功能

- PDF / MD 文档解析
- 文档内容切分与知识片段管理
- MinIO 文件对象存储
- MongoDB 文档元数据与问答记录存储
- Milvus 向量入库与相似度检索
- Neo4j 实体关系存储
- LangGraph 问答流程编排
- FastAPI 接口封装

## 项目亮点

- 基于 MinerU 将非结构化 PDF 文档解析为文本、图片和结构化内容
- 基于 Milvus 实现知识片段向量检索，支持用户问题相关上下文召回
- 基于 Neo4j 存储实体关系和知识关联信息，为图谱增强问答提供基础
- 基于 MinIO + MongoDB 实现文件资源与业务数据解耦存储
- 基于 LangChain / LangGraph 拆分文档解析、知识检索、上下文拼接和答案生成流程

## 项目结构

```text
.
├── app/                    # 核心业务代码
├── doc/                    # 项目文档
├── prompts/                # Prompt 模板
├── test/                   # 测试脚本
├── docker-compose.yml      # 依赖服务配置
├── main.py                 # 项目入口
├── pyproject.toml          # Python 项目配置
├── uv.lock                 # 依赖锁定文件
└── README.md
## 安装依赖
uv sync

# 启动依赖服务
docker compose up -d

# 启动项目
python main.py


---

## 第五步：README 改完后再提交一次

你在 PyCharm 改完 `README.md` 后，执行：

```powershell
查看改了哪些文件   git status  git diff  查看具体改动，可选但推荐  按q退出
推荐你指定文件添加   eg git add app/import_process/agent/main_graph.py
                    git add app/import_process/agent/nodes/node_import_milvus.py
                    git add app/import_process/api/
                    git add test/test_fastapi.py
                    
确认暂存区   git status  
确认没有
.env
data/
models/
logs/
*.pt
*.pth
*.bin
*.safetensors
node_modules/

提交
git commit -m "Update import process"

上传到 GitHub
git push


以后你修改完代码，就执行：

git status
git add .
git status
git diff --cached --name-only
git commit -m "Update code"
git push


如果只是删除了文件

删除文件后执行：

git status
git add -u
git commit -m "Remove unused files"
git push