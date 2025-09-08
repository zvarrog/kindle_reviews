import requests

url = "http://localhost:8000/predict"
resp = requests.post(
    url,
    json={
        "texts": [
            "Loved this Kindle!",
            "Not bad.",
            "Terrible experience",
        ]
    },
)
print(resp.status_code, resp.json())
