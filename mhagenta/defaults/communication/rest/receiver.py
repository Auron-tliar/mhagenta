from fastapi import FastAPI, APIRouter

app = FastAPI()


def add_router(router: APIRouter) -> None:
    app.include_router(router)
