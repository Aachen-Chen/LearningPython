from django.shortcuts import render

# Create your views here.
def rego(request):
    return render(request, 'rego.html')


