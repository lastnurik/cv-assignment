"""Local-only HTTP application. Upload bytes are never written to disk."""
import argparse
import io
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from urllib.parse import urlparse
from PIL import Image,UnidentifiedImageError
from .preprocessing import open_rgb
from .inference import KerasEngine

MAX_BYTES=10*1024*1024
MAX_PIXELS=16_000_000


def decode_upload(payload):
    if not payload or len(payload)>MAX_BYTES:raise ValueError('Choose an image smaller than 10 MB.')
    with Image.open(io.BytesIO(payload)) as probe:
        if probe.width*probe.height>MAX_PIXELS:raise ValueError('Image exceeds 16 megapixels.')
        if probe.format not in ('JPEG','PNG','WEBP'):raise ValueError('Use JPEG, PNG or WebP.')
    return open_rgb(io.BytesIO(payload))


def handle_prediction(payload,engine):
    try:im=decode_upload(payload)
    except (ValueError,UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
        return 400,{'error':str(exc) if isinstance(exc,ValueError) else 'Cannot decode image; use a valid JPEG, PNG or WebP.'}
    try:return 200,engine.predict(im)
    except RuntimeError as exc:return 503,{'error':str(exc)}
    except Exception:return 500,{'error':'Inference failed. Check the server runtime and model bundle; no prediction was produced.'}


def handler_factory(engine,web_root):
    web_root=Path(web_root)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass  # No image data or request logs persisted.
        def send(self,status,payload,content_type='application/json'):
            body=json.dumps(payload,allow_nan=False).encode() if content_type=='application/json' else payload
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            path=urlparse(self.path).path
            if path=='/api/status':return self.send(200,engine.status())
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','application/javascript'),'/style.css':('style.css','text/css')}
            if path not in files:return self.send(404,{'error':'Not found'})
            name,mime=files[path];return self.send(200,(web_root/name).read_bytes(),mime)
        def do_POST(self):
            if urlparse(self.path).path!='/api/predict':return self.send(404,{'error':'Not found'})
            origin=self.headers.get('Origin')
            if origin and urlparse(origin).netloc!=self.headers.get('Host'):
                return self.send(403,{'error':'Use this local app to upload images.'})
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:return self.send(400,{'error':'Invalid content length'})
            if not 0<length<=MAX_BYTES:return self.send(413,{'error':'Choose an image smaller than 10 MB.'})
            self.connection.settimeout(30)
            try:payload=self.rfile.read(length)
            except TimeoutError:return self.send(408,{'error':'Upload timed out.'})
            status,result=handle_prediction(payload,engine);self.send(status,result)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8765);p.add_argument('--bundle',type=Path)
    p.add_argument('--allow-legacy',action='store_true',help='Explicit historical A1 diagnostic; NOT a final A2 model')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    engine=KerasEngine(root,a.bundle,a.allow_legacy)
    try:
        server=HTTPServer(('127.0.0.1',a.port),handler_factory(engine,root/'web'))
    except OSError as exc:
        raise SystemExit(f'Cannot start local server: {exc}. Run this command in a local terminal with loopback networking enabled.') from exc
    print(f'Local app: http://127.0.0.1:{a.port}',flush=True)
    print(engine.label,flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
