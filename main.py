from app.import_process.api.import_server import app


def main():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8009)
