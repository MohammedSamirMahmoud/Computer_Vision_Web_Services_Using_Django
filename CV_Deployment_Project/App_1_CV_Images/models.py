from django.db import models

# Create your models here.

from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone




class Feedback(models.Model):
    #customer_name = models.CharField(max_length=120)
    #product = models.ForeignKey(Product, on_delete=models.CASCADE)
    details = models.TextField(blank=True)
    date = models.DateTimeField(default=timezone.now)


