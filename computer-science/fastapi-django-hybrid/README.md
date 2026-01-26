# FastAPI + Django Hybrid Practice

This project mounts a FastAPI app under `/api` while keeping standard Django routes at `/`.
It includes a shared Product catalog used by both frameworks.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (ASGI)

```bash
uvicorn hybrid_project.asgi:application --reload
```

- Django products: http://127.0.0.1:8000/products/
- Django login: http://127.0.0.1:8000/accounts/login/
- FastAPI health: http://127.0.0.1:8000/api/health
- FastAPI products: http://127.0.0.1:8000/api/products

## Django manage.py

You can still use `manage.py` for migrations or createsuperuser:

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

## Auth note

FastAPI endpoints require a Django session cookie. Login via Django first,
then call the API from the same browser or with the session cookie.
