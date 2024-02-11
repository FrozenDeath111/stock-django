from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("save-to-database", views.save_data),
    path("add-data", views.add_data, name="add_data"),
    path("prediction-data", views.pred_data, name="prediction_data"),
    path("edit-data/<int:stock_id>", views.edit_data, name="edit_data"),
    path("delete-data/<int:stock_id>", views.delete_data, name="delete_data"),
    path("json", views.index_json, name="index_json"),
    path("csv", views.index_csv, name="index_csv"),
]
