from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "loan_amnt": 10000,
        "annual_inc": 60000,
        "dti": 15,
        "open_acc": 5
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    assert "probability_of_default" in body
    assert body["decision"] in ["APPROVE", "REVIEW", "REJECT"]
