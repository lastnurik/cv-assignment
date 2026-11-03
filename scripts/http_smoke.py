"""Bounded loopback HTTP check using historical weights; no browser/camera claim."""
import json
from pathlib import Path
import threading
from http.server import HTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from gesture_cv.inference import KerasEngine
from gesture_cv.webapp import handler_factory
from gesture_cv.dataset import write_json
from gesture_cv import CLASS_NAMES


def main():
    results = []
    for legacy in (False, True):
        engine = KerasEngine('.', allow_legacy=legacy)
        server = HTTPServer(('127.0.0.1', 0), handler_factory(engine, 'web'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        def request(path, data=None):
            try:
                with urlopen(Request(base+path, data=data), timeout=60) as r:
                    return r.status, r.read()
            except HTTPError as e:
                return e.code, e.read()
        try:
            assert request('/')[0] == 200
            assert request('/api/status')[0] == 200
            assert request('/api/predict', b'invalid')[0] == 400
            for label in CLASS_NAMES:
                path = sorted(Path('dataset', label).glob('*.jpg'))[0]
                code, body = request('/api/predict', path.read_bytes())
                result = json.loads(body)
                assert code == (200 if legacy else 503), result
                assert ('prediction' in result) == legacy
                results.append({'legacy': legacy, 'sample': str(path), 'status': code,
                                'prediction': result.get('prediction')})
        finally:
            server.shutdown(); server.server_close(); thread.join()
    write_json('artifacts/qa/http_smoke.json', {'passed': True, 'results': results,
               'scope': 'Loopback HTTP only; historical model; no browser or camera test'})

if __name__ == '__main__':
    main()
