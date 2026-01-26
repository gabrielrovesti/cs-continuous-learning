from django.db import models


class Product(models.Model):
    name = models.CharField(max_length=120)
    price_eur = models.DecimalField(max_digits=8, decimal_places=2)
    stock = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.name} ({self.price_eur} EUR)"
