from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os
#アドバイスを受け修正した箇所
import numpy as np
from tensorflow.keras.applications.vgg16 import decode_predictions


def predict(request):
    if request.method == "GET":
        form = ImageUploadForm()
        return render(request, "home.html", {'form': form})

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data["image"]
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = img_array / 255

            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)

            # ここから予測を行います
            result = model.predict(img_array)
            decoded_predictions = decode_predictions(result, top=5)[0]
            top_5_predictions = [(pred[1], pred[2]) for pred in decoded_predictions]
            
            img_data = request.POST.get('img_data')
            return render(request, "home.html", {'form': form, 'top_5_predictions': top_5_predictions, 'img_data': img_data})
    
    else:
        form = ImageUploadForm()

    return render(request, "home.html", {'form': form})



