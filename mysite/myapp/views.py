from django.shortcuts import render

from .models import Article

# Create your views here.
def index(request):
    article = Article.objects.get(pk=1)
    return render(request, "index.html", {"article":article})