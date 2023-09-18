from django import forms

class TimestampForm(forms.Form):
    timestamp = forms.DateTimeField(
        label='Enter a Future Timestamp (e.g., "2023-09-18 14:00:00")',
        input_formats=['%Y-%m-%d %H:%M:%S'],
        widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD HH:MM:SS', 'class': 'input'}),
    )

