from django.contrib import admin
from django.urls import path
from app1.views import SkillSearchView, AddSkillView,DeleteSkillView  # Import both views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('search/', SkillSearchView.as_view(), name='skill_search'),  # Route to the Skill Search view
    path('addskill/', AddSkillView.as_view(), name='add_skill'),  # Route to the Add Skill view (corrected name)
    path('delete_skill/<str:skill_id>/', DeleteSkillView.as_view(), name='delete_skill'),]
