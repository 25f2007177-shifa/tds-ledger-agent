
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TDS Ledger Agent")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"status": "Ledger Agent is running"}


@app.post("/")
def answer_question(request: QuestionRequest):
    # Temporary response; we will add the ledger logic next.
    return {"answer": "Ledger Agent is being set up."}
