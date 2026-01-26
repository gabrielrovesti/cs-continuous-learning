import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hybrid_project.settings")

from django.core.asgi import get_asgi_application
from fastapi_app.main import app as fastapi_app
from starlette.applications import Starlette
from starlette.routing import Mount

django_asgi_app = get_asgi_application()

application = Starlette(
    routes=[
        Mount('/api', app=fastapi_app),
        Mount('/', app=django_asgi_app),
    ]
)
