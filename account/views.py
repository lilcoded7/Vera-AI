from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import *
from .forms import *


# Create your views here.
def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            password = form.cleaned_data['password1'] 
            customer = Customer.objects.create(user=user, password=password)
            return redirect('login')
        else:
            messages.error(request, 'Registration failed. Please enter correct details.')
    else:
        form = UserRegisterForm()

    return render(request, 'auths/register.html', {'form': form})


def login_view(request):
    form = LoginForm()
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                print('-0-0-0-0-0-0-0-0-0-')
                login(request, user)
                return redirect('connect/bms/api/')
            else:
                messages.error(request, 'Invalid credentials, login failed!')
        else:
            form = LoginForm()
    return render(request, 'auths/login.html', {'form': form})
