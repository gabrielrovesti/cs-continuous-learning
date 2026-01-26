from decimal import Decimal

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from .models import Product

def home(request: HttpRequest) -> HttpResponse:
    return redirect("products")


@login_required
def product_list(request: HttpRequest) -> HttpResponse:
    error = ""
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        price = request.POST.get("price_eur", "").strip()
        stock = request.POST.get("stock", "").strip()
        if name and price:
            try:
                price_value = Decimal(price)
                stock_value = int(stock) if stock else 0
            except (ValueError, ArithmeticError):
                error = "Prezzo o stock non validi."
            else:
                Product.objects.create(
                    name=name,
                    price_eur=price_value,
                    stock=stock_value,
                )
                return redirect("products")
        else:
            error = "Nome e prezzo sono obbligatori."

    query = request.GET.get("q", "").strip()
    min_price = request.GET.get("min_price", "").strip()
    max_price = request.GET.get("max_price", "").strip()
    in_stock = request.GET.get("in_stock") == "1"

    products = Product.objects.order_by("-created_at")
    if query:
        products = products.filter(Q(name__icontains=query))
    if min_price:
        try:
            products = products.filter(price_eur__gte=Decimal(min_price))
        except (ValueError, ArithmeticError):
            error = "Filtro prezzo minimo non valido."
    if max_price:
        try:
            products = products.filter(price_eur__lte=Decimal(max_price))
        except (ValueError, ArithmeticError):
            error = "Filtro prezzo massimo non valido."
    if in_stock:
        products = products.filter(stock__gt=0)

    paginator = Paginator(products, 10)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(
        request,
        "products.html",
        {
            "products": page_obj.object_list,
            "page_obj": page_obj,
            "error": error,
            "query": query,
            "min_price": min_price,
            "max_price": max_price,
            "in_stock": in_stock,
        },
    )


@login_required
def product_edit(request: HttpRequest, product_id: int) -> HttpResponse:
    product = get_object_or_404(Product, id=product_id)
    error = ""
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        price = request.POST.get("price_eur", "").strip()
        stock = request.POST.get("stock", "").strip()
        if name and price:
            try:
                product.name = name
                product.price_eur = Decimal(price)
                product.stock = int(stock) if stock else 0
            except (ValueError, ArithmeticError):
                error = "Prezzo o stock non validi."
            else:
                product.save()
                return redirect("products")
        else:
            error = "Nome e prezzo sono obbligatori."

    return render(
        request,
        "product_form.html",
        {"product": product, "error": error},
    )


@login_required
def product_delete(request: HttpRequest, product_id: int) -> HttpResponse:
    product = get_object_or_404(Product, id=product_id)
    if request.method == "POST":
        product.delete()
        return redirect("products")
    return render(request, "product_confirm_delete.html", {"product": product})
