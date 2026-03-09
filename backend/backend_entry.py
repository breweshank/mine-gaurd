from fastapi import FastAPI
import uvicorn
from app.routes.analyze import router

app = FastAPI()

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("backend_entry:app", host="127.0.0.1", port=5051, reload=True)
