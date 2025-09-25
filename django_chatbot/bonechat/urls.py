from django.urls import path
from . import views

urlpatterns = [
    path('', views.login, name='login'),
    path('lchatbot/', views.chatbot, name='chatbot'),
    path('chatbot',  views.chatbot),
    path('register/', views.register, name='register'),
    path('register',  views.register),
    path('logout/', views.logout, name='logout'),
    path('logout',  views.logout),
    path('folder/', views.folder, name='folder'),
    path('folder',  views.folder),
    path('histories/', views.histories, name='histories'),
    path('histories',  views.histories),
]
