from django.urls import path
import sio.views

urlpatterns = [
    path('', sio.views.home),
]
