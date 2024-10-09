from django.contrib import admin
from django.urls import path
from app1.views import SkillSearchView  # Import the view here

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', SkillSearchView.as_view(), name='hello_world'),  # Route to the Hello World view

]


