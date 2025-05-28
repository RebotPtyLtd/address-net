import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import threading
from http.server import HTTPServer

import requests
import pytest

import http_endpoint


def test_http_endpoint(monkeypatch):
    def fake_predict(address, model_dir=None):
        return {'street_name': 'HIGH', 'street_type': 'STREET'}, 0.95

    monkeypatch.setattr(http_endpoint, '_predict_address', fake_predict)

    server = HTTPServer(('localhost', 0), http_endpoint.AddressHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    try:
        url = f'http://localhost:{server.server_port}/predict'
        resp = requests.get(url, params={'address': '10 High Street'})
        assert resp.status_code == 200
        data = resp.json()
        assert data['input'] == '10 High Street'
        assert data['reformatted'] == 'HIGH STREET'
        assert data['confidence'] == 0.95
    finally:
        server.shutdown()
        thread.join()

