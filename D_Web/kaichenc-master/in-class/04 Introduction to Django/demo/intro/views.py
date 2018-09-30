from django.http import HttpResponse
from django.shortcuts import render

# The action for the 'intro/hello-world' route.
def hello_world_simple(request):
    # Returns an HTTPResponse constructed from an HTML string
    response_string = """
<!DOCTYPE HTML>
<html>
<head>
  <meta charset="utf-8">
  <title>Hello World</title>
</head>

<body>
<h1>Hello, world!</h1>
</body>
</html>
"""
    return HttpResponse(response_string)

# The action for the 'intro/hello-world-2' route.
def hello_world_template(request):
    # render takes: (1) the request,
    #               (2) the name of the view to generate, and
    #               (3) a dictionary of name-value pairs of data to be
    #                   available to the view template.
    return render(request, 'generic-hello.html-template', {})


# The action for the 'intro/hello.html' route.
def hello(request):
    # Creates a Python dictionary that will be used to make name-value
    # pairs available to the view.
    context = {}
    context['person_name'] = ''

    # Retrieves the name from the request if the 
    # 'username' HTTP GET parameter is present.
    if 'username' in request.GET:
        context['person_name'] = request.GET['username']

    # Renders the response using the greet.html template and the
    # key-value pairs in the context dictionary.
    return render(request, 'greet.html', context)

