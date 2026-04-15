import requests

res = requests.post(
    "http://localhost:8000/query",
    json={"query": "Who has the highest deposit?"}
)

print("Status Code:", res.status_code)
print("Response JSON:")
print(res.json())