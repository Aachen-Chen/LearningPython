from django.urls import path, include
import sio.views

urlpatterns = [
    path('sio/', include('sio.urls')),
    path('', sio.views.home),
]
