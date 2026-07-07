`# RAG 知识库问答系统

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
docker compose up -d#只会启动当前目录下的docker-compose.yml

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


看 Milvus 向量数据                                                               
                                                                                      
  用 Attu。                                                                           
                                                                                      
  浏览器打开你的 Attu 地址，一般是：                                                                                                                            
  http://127.0.0.1:8000  
                                                               
 Docker 里 Attu 映射的端口。                                                                                                                 
  进去后连接 Milvus：                                                                                                                                                
  Milvus Address: 127.0.0.1:19530 

看 MinIO 文件数据                                                                
                                                                                      
  浏览器打开：                                                                        
                                                                                      
  http://127.0.0.1:9001                                                               
                                                                                      
  账号密码：                                                                          
                                                                                      
  minioadmin                                                                          
  minioadmin   

直接看/管理 MinIO 里的文件                                                       
                                                                                      
  这个是对象存储后台，不是业务上传页面。                                              
                                                                                      
  地址是：                                                                            
                                                                                      
  http://127.0.0.1:9001                                                               
                                                                                      
  账号密码：                                                                          
                                                                                      
  minioadmin                                                                          
  minioadmin 

---

## 本地导入、查看和搜索流程

这一节按实际操作顺序记录：先启动 Docker 依赖，再上传文件导入知识库，然后查看导入结果，最后启动查询服务做搜索问答。

### 1. 启动 Docker 依赖服务

在项目根目录执行：

```powershell
cd D:\Python基础\dataset_rag
docker compose up -d
```

这条命令只会读取当前目录下的 `docker-compose.yml`，启动本项目定义的服务：

```text
dataset_rag-etcd-1
dataset_rag-minio-1
dataset_rag-milvus-standalone-1
dataset_rag-mongo-1
```

检查容器是否启动：

```powershell
docker ps
```

主要端口：

```text
Milvus: 127.0.0.1:19530
MongoDB: 127.0.0.1:27017
MinIO API: 127.0.0.1:9000
MinIO Console: http://127.0.0.1:9001
```

### 2. 启动导入服务

导入服务负责上传 PDF/MD，并执行解析、切分、向量化、写入 Milvus。

```powershell
cd D:\Python基础\dataset_rag
D:\anaconda\envs\agent\python.exe main.py
```

看到类似下面日志，表示启动成功：

```text
Uvicorn running on http://127.0.0.1:8009
```

浏览器打开上传页面：

```text
http://127.0.0.1:8009/import
```

上传接口实际是：

```text
POST http://127.0.0.1:8009/upload
```

### 5. 查看 MinIO 文件
MinIO 是对象存储，用来看上传文件、图片、Markdown 等对象数据。
浏览器打开：
```text
http://127.0.0.1:9001
```
账号密码：
```text
minioadmin
minioadmin
```

进入后查看 bucket：

```text
knowledge-base-files
```

如果 `9001` 打不开，先执行：

```powershell
docker compose up -d minio
```

### 6. 查看 MongoDB 聊天历史

用 MongoDB Compass 连接：

```text
mongodb://127.0.0.1:27017/
```

项目数据库：

```text
kb002
```

常见集合：

```text
chat_message
```

这里保存查询过程中的用户问题、助手回答、改写后的 query、关联商品名等记录。

### 7. 查看 Milvus 向量数据

Milvus 里保存文档切片向量和商品名向量。

使用 Attu 连接 Milvus：

```text
Milvus Address: 127.0.0.1:19530
```

重点查看 collection：

```text
kb_chunks
kb_item_names
```

含义：

```text
kb_chunks: 文档切片向量
kb_item_names: 商品名向量
```

如果上传日志里出现：

```text
成功插入5
```

一般表示 `kb_chunks` 中插入了 5 条切片数据。

### 8. 启动查询服务并搜索

查询服务负责聊天问答、商品名确认、向量检索和答案输出。

```powershell
cd D:\Python基础\dataset_rag
D:\anaconda\envs\agent\python.exe -m uvicorn app.query_process.api.query_server:app --host 127.0.0.1 --port 8008
```

浏览器打开聊天页面：

```text
http://127.0.0.1:8008/chat.html
```

健康检查：

```text
http://127.0.0.1:8008/health
```

聊天接口实际是：

```text
POST http://127.0.0.1:8008/query
```

### 9. 常见端口总结

```text
8009: 导入服务，上传页面 /import
8008: 查询服务，聊天页面 /chat.html
19530: Milvus
27017: MongoDB
9001: MinIO 控制台
9000: MinIO API
```


