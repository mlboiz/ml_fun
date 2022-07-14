# ML FUN!
## LAYOUT IS STILL WIP 🚧🚧🚧
So basically the premise of this repo is to have fun with some Deep Learning applications.

At this moment you can have fun with:
- text generation (courtesy of GPT2 and HuggingFace🙏)
- style transfer
- deep dream

## How to run it?
1. Install Python 3.8
2. Create venv, open it and then run `pip install -r requirements.txt`
3. Just run `python run.py` and enjoy! Your app is running on `localhost:8050`.

### How to enjoy photo fun?
Remember to allow capturing video, then just take a photo and perform style transfer or deep dreaming!

### Deep Dream fun
In terms of Deep Dream we've picked 3 models: 
- InceptionV3
- EfficientNet V2 B2
- VGG16

What is more for every model we picked some layers that had best results for dreaming (you can even combine layers!).

Best params for VGG16:
```
step_size = 1.
steps = 15
```

For rest of the models:
```
step_size = 1e-2
steps = 50
```

And for all of them:
```
num_of_octaves = 4
octave_scale = 1.4
```