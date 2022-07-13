from django.db import models

# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=30)
    descr = models.CharField(max_length=50)
    date_published = models.DateField(auto_now=False, auto_now_add=False)
    image = models.ImageField(upload_to='images/')
