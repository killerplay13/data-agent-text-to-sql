from fastapi import FastAPI
from app.api.query import router as query_router


app = FastAPI(
    title="Data Agent Text-to-SQL API",
    version="0.1.0"
)

app.include_router(query_router)


@app.get("/")
def root():
    return {"message": "Data Agent Text-to-SQL API is running"}