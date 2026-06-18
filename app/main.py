from fastapi import FastAPI, HTTPException

app = FastAPI(title="Video Search API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(q: str, k: int = 5) -> list[dict]:
    raise HTTPException(status_code=501, detail="Not implemented")
