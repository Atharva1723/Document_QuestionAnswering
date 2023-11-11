# AUTHOR : TECHMAHINDRA MAKERS LAB #

# Import necessary modules from Django admin
from django.contrib import admin
from simple_history.admin import SimpleHistoryAdmin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import bot_details, bot_admin, entellio_UserAdmin, entellio_Bot, \
    corpus_configuration

# Define a custom admin class for 'corpus_configuration'
class corpus_config(admin.ModelAdmin):
    list_display = ('botid', 'language', 'stopwords', 'sp_characters')

# Define a custom UserAdmin class
class CustomUserAdmin(UserAdmin):
    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()

# Register the models and their respective admin classes
admin.site.register(bot_details, entellio_Bot)
admin.site.register(bot_admin, entellio_UserAdmin)
admin.site.register(corpus_configuration, corpus_config)
# admin.site.unregister(User)
# admin.site.register(User, CustomUserAdmin)

