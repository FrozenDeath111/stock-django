from django import forms
from django.forms import ModelForm
from .models import Stock


class StockForm(ModelForm):
    class Meta:
        model = Stock
        fields = "__all__"
        labels = {
            'date': 'Enter Date (Format -> YYYY-MM-DD)',
            'trade_code': 'Enter Trade Code',
            'high': 'Enter High',
            'low': 'Enter Low',
            'open': 'Enter Open',
            'close': 'Enter Close',
            'volume': 'Enter Volume',
        }
        widgets = {
            'date': forms.DateInput(attrs={'class': 'form-control'}),
            'trade_code': forms.TextInput(attrs={'class': 'form-control'}),
            'high': forms.NumberInput(attrs={'class': 'form-control'}),
            'low': forms.NumberInput(attrs={'class': 'form-control'}),
            'open': forms.NumberInput(attrs={'class': 'form-control'}),
            'close': forms.NumberInput(attrs={'class': 'form-control'}),
            'volume': forms.NumberInput(attrs={'class': 'form-control'}),
        }
