from django.contrib import admin
from django.urls import path
from app1.views import AppliedSkillSearchView,ApprovedSkillSearchView # Import both views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('search/', AppliedSkillSearchView.as_view(), name='skill_search'),  # Route to the Skill Search view

    path('search/<int:skill_id>/', AppliedSkillSearchView.as_view(), name='delete_skill'),

    path("recommend_skills/", ApprovedSkillSearchView.as_view(), name="recommend_skills"),
    path('recommend_skills/<int:skill_id>/', ApprovedSkillSearchView.as_view(), name='delete_rec_skill'),

]
