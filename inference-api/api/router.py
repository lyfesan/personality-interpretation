from fastapi import APIRouter
from api.endpoints import system, predict

api_router = APIRouter()

api_router.include_router(system.router)
api_router.include_router(predict.router)
