import requests

url = "http://127.0.0.1:5000/predict"
image_path = "../data/train/phase3_bursting/3.jpg"

with open(image_path, "rb") as f:
    files = {"image": f}
    r = requests.post(url, files=files)

print(r.json())
