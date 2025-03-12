from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from semantic_search.search import SemanticSearch

app = FastAPI()
search_engine = None

class IndexRequest(BaseModel):
    csv_path: str
    template: str
    backend: str = "cosine"
    model: str = None
    model_name: str = "all-MiniLM-L6-v2"

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/index/")
def index_data(request: IndexRequest):
    global search_engine
    try:
        search_engine = SemanticSearch(backend=request.backend, model_path=request.model, model_name=request.model_name)
        search_engine.build_index_from_csv(request.csv_path, request.template)
        return {"message": "Index built successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search/")
def search_data(request: SearchRequest):
    if search_engine is None:
        raise HTTPException(status_code=400, detail="Index not built yet")
    try:
        distances, indices = search_engine.query(request.query, request.top_k)
        return {"distances": distances.tolist(), "indices": indices.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
