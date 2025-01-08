from django.shortcuts import render
from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()


# Create your views here.
