from django.urls import path, include
from sio.views import home

urlpatterns = [
    path('sio/', include('sio.urls')),
    path('', home),
]
