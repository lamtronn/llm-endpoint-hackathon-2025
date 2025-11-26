"""
API v1
"""

from fastapi import APIRouter
from .endpoints import blip2, sam, attco, autopet

api_router = APIRouter()

api_router.include_router(blip2.router, prefix="/blip2", tags=["BLIP2"])
api_router.include_router(sam.router, prefix="/sam", tags=["SAM"])
api_router.include_router(attco.router, prefix="/attco", tags=["AttCo"])
api_router.include_router(autopet.router, prefix="/autopet", tags=["AutoPET"])
