from django.shortcuts import render

from .models import Article
from django.views.generic import ListView

# Create your views here.
def index(request):
    list = Article.objects.all()
    return render(request, "index.html", {"list":list})
