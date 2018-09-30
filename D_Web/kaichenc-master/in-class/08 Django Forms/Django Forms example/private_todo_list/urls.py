from django.urls import path
from private_todo_list import views
from django.contrib.auth.views import LoginView, logout_then_login

urlpatterns = [
    path('', views.home, name='home'),
    path('add-item', views.add_item, name='add'),
    path('delete-item/<int:id>', views.delete_item, name='delete'),
    # Route for built-in authentication with our own custom login page
    path('login', LoginView.as_view(template_name='private-todo-list/login.html'), name='login'),
    # Route to logout a user and send them back to the login page
    path('logout', logout_then_login, name='logout'),
    path('register', views.register, name='register'),
]
