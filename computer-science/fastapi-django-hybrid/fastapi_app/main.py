import os
from decimal import Decimal
from typing import Any

import django
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.sessions.backends.db import SessionStore
from django.db.models import Q
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hybrid_project.settings")
django.setup()

from django_app.models import Product

app = FastAPI(title="Hybrid Practice API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _get_current_user(request: Request) -> Any:
    session_key = request.cookies.get(settings.SESSION_COOKIE_NAME)
    if not session_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session = SessionStore(session_key=session_key)
    data = session.load()
    user_id = data.get("_auth_user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user_model = get_user_model()
    try:
        user = user_model.objects.get(id=user_id)
    except user_model.DoesNotExist as exc:
        raise HTTPException(status_code=401, detail="Not authenticated") from exc

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User inactive")

    return user


class ProductCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    price_eur: Decimal
    stock: int = Field(default=0, ge=0)


class ProductUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    price_eur: Decimal | None = None
    stock: int | None = Field(default=None, ge=0)


def _serialize_product(product: Product) -> dict:
    return {
        "id": product.id,
        "name": product.name,
        "price_eur": str(product.price_eur),
        "stock": product.stock,
        "created_at": product.created_at.isoformat(),
    }


@app.get("/me")
def whoami(user: Any = Depends(_get_current_user)) -> dict:
    return {"id": user.id, "username": user.get_username()}


@app.get("/products")
def list_products(
    user: Any = Depends(_get_current_user),
    q: str | None = None,
    min_price: Decimal | None = None,
    max_price: Decimal | None = None,
    in_stock: bool = False,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    products = Product.objects.order_by("-created_at")
    if q:
        products = products.filter(Q(name__icontains=q))
    if min_price is not None:
        products = products.filter(price_eur__gte=min_price)
    if max_price is not None:
        products = products.filter(price_eur__lte=max_price)
    if in_stock:
        products = products.filter(stock__gt=0)

    total = products.count()
    items = products[offset : offset + limit]
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [_serialize_product(product) for product in items],
    }


@app.get("/products/{product_id}")
def get_product(
    product_id: int,
    user: Any = Depends(_get_current_user),
) -> dict:
    try:
        product = Product.objects.get(id=product_id)
    except Product.DoesNotExist as exc:
        raise HTTPException(status_code=404, detail="Product not found") from exc
    return _serialize_product(product)


@app.post("/products", status_code=201)
def create_product(
    payload: ProductCreate,
    user: Any = Depends(_get_current_user),
) -> dict:
    product = Product.objects.create(
        name=payload.name.strip(),
        price_eur=payload.price_eur,
        stock=payload.stock,
    )
    return _serialize_product(product)


@app.patch("/products/{product_id}")
def update_product(
    product_id: int,
    payload: ProductUpdate,
    user: Any = Depends(_get_current_user),
) -> dict:
    try:
        product = Product.objects.get(id=product_id)
    except Product.DoesNotExist as exc:
        raise HTTPException(status_code=404, detail="Product not found") from exc

    if payload.name is not None:
        product.name = payload.name.strip()
    if payload.price_eur is not None:
        product.price_eur = payload.price_eur
    if payload.stock is not None:
        product.stock = payload.stock

    product.save()
    return _serialize_product(product)


@app.delete("/products/{product_id}", status_code=204)
def delete_product(
    product_id: int,
    user: Any = Depends(_get_current_user),
) -> None:
    deleted = Product.objects.filter(id=product_id).delete()[0]
    if not deleted:
        raise HTTPException(status_code=404, detail="Product not found")
    return None
