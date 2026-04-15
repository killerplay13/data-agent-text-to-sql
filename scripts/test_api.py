import requests

res = requests.post(
    "http://127.0.0.1:8000/query",
    json={"query": "Who has the highest deposit?"}
)

print(res.status_code)
print(res.json())