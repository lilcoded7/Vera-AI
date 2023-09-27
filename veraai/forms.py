from django import forms

class EnergyPredictionForm(forms.Form):
    
    building_name = forms.CharField(
        label='Building Name', 
        max_length=100,
        widget=forms.TextInput(
            attrs={
                'placeholder': 'Building name',
                'class': 'input'
            }
        )
    )
    
    future_date = forms.DateTimeField(
        label='Future Date and Time',
        widget=forms.DateTimeInput(
            attrs={
                'type': 'datetime-local',
                'class': 'input',  # Add your desired CSS class for styling
                'style': 'color: red;'  # Add any additional CSS styles
            }
        )
    )
