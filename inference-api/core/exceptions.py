from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

async def custom_404_handler(request: Request, exc: StarletteHTTPException):
    """Custom handler to format 404 errors cleanly."""
    if exc.status_code == 404:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Endpoint not found", 
                "path": request.url.path,
                "message": "Please check the URL or visit /docs for available endpoints."
            }
        )
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
