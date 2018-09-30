from django.urls import path, include
import shared_todo_list.views

urlpatterns = [
    path('shared-todo-list/', include('shared_todo_list.urls')),
    path('', shared_todo_list.views.home),
]
