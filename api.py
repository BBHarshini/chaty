from fastapi import FastAPI
from pydantic import BaseModel
from chat import get_answer


app = FastAPI()


class Sentence(BaseModel):
    content: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/messages")
def question(q: str):
    answer = get_answer(q)
    return {"answer": answer}
