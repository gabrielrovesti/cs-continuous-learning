from django.urls import path
from .views import home, product_delete, product_edit, product_list

urlpatterns = [
    path("", home, name="home"),
    path("products/", product_list, name="products"),
    path("products/<int:product_id>/edit/", product_edit, name="product_edit"),
    path("products/<int:product_id>/delete/", product_delete, name="product_delete"),
]
