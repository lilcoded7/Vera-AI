from django import forms

class FuturePredictionForm(forms.Form):
    future_date = forms.DateField(label='Select a Future Date', widget=forms.DateInput(attrs={'type': 'date'}))
    future_time = forms.TimeField(label='Select a Future Time', widget=forms.TimeInput(attrs={'type': 'time'}))
