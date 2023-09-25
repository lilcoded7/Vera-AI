from django import forms 
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm



class UserRegisterForm(UserCreationForm):
    username = forms.CharField(
        required=True, 
        widget=forms.TextInput(
            attrs={
                'placeholder': 'Username',
                'class': 'form-control'
            }
        )
    )

    email = forms.EmailField(
        required=True, 
        widget=forms.TextInput(
            attrs={
                'placeholder': 'Email',
                'class': 'form-control'
            }
        )
    )

    password1 = forms.CharField(
        required=True, 
        widget=forms.PasswordInput(
            attrs={
                'placeholder': 'Password',
                'class': 'form-control'
            }
        )
    )

    password2 = forms.CharField(
        required=True, 
        widget=forms.PasswordInput(
            attrs={
                'placeholder': 'Confirm Password',
                'class': 'form-control'
            }
        )
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class LoginForm(forms.Form):
    username = forms.CharField(
        required=True, widget=forms.TextInput(
            attrs={
                'placeholder':'enter username',
                'class':'form-control'
            }
        )
    )

    password = forms.CharField(
        required=True, widget=forms.TextInput(
            attrs={
                'type'       : 'password',
                'placeholder':'enter password',
                'class':'form-control'
            }
        )
    )






