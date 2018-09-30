from django.urls import path

import intro.views

urlpatterns = [
    path('intro/hello-world', intro.views.hello_world_simple),
    path('intro/hello-world-2', intro.views.hello_world_template),
    path('intro/hello.html', intro.views.hello),
]
