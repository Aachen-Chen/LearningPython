
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views


app_name = 'grumblr'
urlpatterns = [
    path('', views.home, name="home"),

    path('login/', auth_views.LoginView.as_view(template_name='grumblr/login.html'), name='login'),

    path('logout/', auth_views.logout_then_login, name='logout'),

    path('register/', views.rego, name='rego'),

    path('profile/<username>/', views.profile, name='profile'),

    path('post/', views.post, name='post'),
]