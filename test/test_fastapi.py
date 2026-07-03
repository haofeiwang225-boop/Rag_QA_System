from fastapi import FastAPI
from openai import BaseModel

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.get("/items/")
def read_items(skip: int = 0, limit: int = 10):
    print(skip, limit)
    return {"skip": skip, "limit": limit}

# 定义数据模型
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

# 使用 Header 和 Cookie 类型注解获取请求头和 Cookie 数据。
# POST 请求接收 JSON 数据
@app.post("/items/")
def create_item(item: Item):
    # item 已经是验证过的 Item 对象
    # 如果客户端传来的 price 是字符串 "abc"，FastAPI 会自动报错
    return {"item_name": item.name, "item_price": item.price}

from fastapi import Header, Cookie
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/")
def read_item(user_agent: str = Header(None), session_token: str = Cookie(None)):
    return {"User-Agent": user_agent, "Session-Token": session_token}