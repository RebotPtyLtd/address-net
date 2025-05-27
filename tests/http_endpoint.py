import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse


def _predict_address(address: str, model_dir: str | None = None):
    import tensorflow.compat.v1 as tf
    import tensorflow_estimator as tf_estimator
    from addressnet.predict import labels_list, predict_input_fn
    from addressnet.model import model_fn
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'addressnet', 'pretrained')
    model_dir = os.path.abspath(model_dir)
    estimator = tf_estimator.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)
    result_iter = estimator.predict(predict_input_fn([address]))
    res = next(result_iter)
    class_ids = res['class_ids']
    probs = res['probabilities']
    class_names = [l.replace('_code', '') for l in labels_list]
    class_names = [l.replace('_abbreviation', '') for l in class_names]
    mapping = {}
    conf_vals = []
    for char, cid, prob in zip(address.upper(), class_ids, probs):
        if cid == 0:
            continue
        conf_vals.append(prob[cid])
        cls = class_names[cid - 1]
        mapping[cls] = mapping.get(cls, '') + char
    confidence = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
    return mapping, confidence


_COMPONENT_ORDER = [
    'building_name',
    'flat_type',
    'flat_number',
    'flat_number_suffix',
    'level_type',
    'level_number',
    'number_first',
    'number_last',
    'street_name',
    'street_type',
    'street_suffix',
    'locality_name',
    'state',
    'postcode',
]


def _format_address(parts: dict) -> str:
    ordered = []
    for key in _COMPONENT_ORDER:
        if key in parts:
            ordered.append(str(parts[key]))
    return ' '.join(ordered)


class AddressHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != '/predict':
            self.send_error(404, 'not found')
            return
        qs = parse_qs(parsed.query)
        if 'address' not in qs:
            self.send_error(400, 'address parameter missing')
            return
        address = qs['address'][0]
        mapping, conf = _predict_address(address)
        reformatted = _format_address(mapping)
        payload = {'input': address, 'reformatted': reformatted, 'confidence': conf}
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    port = 8000
    server = HTTPServer(('0.0.0.0', port), AddressHandler)
    print(f'Serving on port {port}')
    server.serve_forever()
