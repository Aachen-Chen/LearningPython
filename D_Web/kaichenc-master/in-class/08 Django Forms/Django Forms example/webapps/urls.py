from django.urls import path, include
from private_todo_list.views import home

urlpatterns = [
    path('private-todo-list/', include('private_todo_list.urls')),
    path('', home),
]
