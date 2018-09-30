from django.urls import path
import shared_todo_list.views

urlpatterns = [
    path('', shared_todo_list.views.home),
    path('add-item', shared_todo_list.views.add_item),
    # Parses integer from URL and uses it as the item_id argument to the action
    path('delete-item/<int:item_id>', shared_todo_list.views.delete_item)
]
