## CovidNet Backend

Covid-19 detection using transfer learning.

## Description
The API uses the `VGG19` convoution neural network, which is trained on a dataset of 21617 images belonging to 2 classes. Number of images used for cross-validation were 6176 and the number of images used for testing were 3089.
The classes (as used in the code) -
```py
labels = {
    0: "negative",
    1: "positive",
}
```

## Usage
- The API can be accessed through the URL - https://covid-net-backend.herokuapp.com/
- To predict an image's class, use the `/predict` endpoint
- For the complete documentation refer to - https://covid-net-backend.herokuapp.com/docs

## Running locally
### To train the model locally -
1. Fork and clone the repository
```
git clone https://github.com/<your_username>/ConvNet-Backend
```
2. Create a new virtual environment
```
python -m venv .venv
```
3. Activate the virtual environment
```
.venv/Scripts/activate
```
4. Run the jupyter in the virtual environment
```
ipython kernel install --user --name=venv
# select the kernel named after your virtual environment in jupyter notebook
```
### To run the API locally-
1. Fork and clone the repository
```
git clone https://github.com/<your_username>/SceneNet-Backend
```
2. Create a new virtual environment
```
python -m venv .venv
```
3. Activate the virtual environment
```
.venv/Scripts/activate
```
4. Install requirements for training (the `Heroku` deployment uses `tensorflow-cpu` and `opencv-python-headless` because of the memory limitations, but you can switch to `tensorflow` and `opencv-python` if you are running this locally)
```
python -m pip install -r requirements.txt
```
5. Fire up the API
```
uvicorn backend.backend:app --reload
```
