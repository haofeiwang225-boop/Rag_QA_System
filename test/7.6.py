import json

# 模拟截图中的接口返回数据
response_data = {
    "isError": False,
    "content": [
        {
            "text": """
            {
                "pages": [
                    {
                        "snippet": "和讯首页 手机和讯 登录注册 股票客户端 Android 股票客户端 iPhone",
                        "hostname": "和讯网",
                        "hostlogo": "https://example.com/hexun-logo.png",
                        "title": "行情中心-和讯网 国内全面的即时行情数据服务中心",
                        "url": "https://quote.hexun.com/"
                    },
                    {
                        "snippet": "数据中心",
                        "hostname": "东方财富网",
                        "hostlogo": "https://example.com/eastmoney-logo.png",
                        "title": "东方财富数据中心",
                        "url": "https://data.eastmoney.com/"
                    }
                ]
            }
            """,
            "type": "text"
        }
    ]
}


#解析 得到pages
text = response_data.get("content")[0].get("text")#字符串
print(type(text))

web_documents = json.loads(text)#转为字典
print(type(web_documents))
print(web_documents)

pages = web_documents.get("pages", [])
print(type(pages))
print(pages)
