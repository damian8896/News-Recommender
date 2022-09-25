from django.db import models

# Create your models here.
class Article(models.Model):
    title = models.CharField(max_length=200)
    descr = models.CharField(max_length=300)
    image = models.ImageField(null=True, blank=True)
    link = models.URLField(default='google.com')
