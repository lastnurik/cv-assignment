import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from gesture_cv.inference import KerasEngine,overlay_heatmap
from gesture_cv.webapp import decode_upload,handle_prediction,handler_factory,MAX_BYTES

class AppContracts(unittest.TestCase):
    def png(self):
        stream=io.BytesIO();Image.new('RGB',(40,30),'red').save(stream,format='PNG');return stream.getvalue()
    def test_valid_image_decode(self):
        self.assertEqual(decode_upload(self.png()).size,(40,30))
    def test_invalid_upload_never_reaches_model(self):
        class MustNotRun:
            def predict(self,im):raise AssertionError('invalid upload reached model')
        self.assertEqual(handle_prediction(b'not an image',MustNotRun())[0],400)
        with self.assertRaises(ValueError):decode_upload(b'x'*(MAX_BYTES+1))
    def test_missing_bundle_produces_no_prediction(self):
        e=KerasEngine(Path('.'))
        code,result=handle_prediction(self.png(),e)
        self.assertEqual(code,503);self.assertNotIn('prediction',result)
        self.assertFalse(e.status()['loaded'])
    def test_missing_tensorflow_produces_no_fake_result(self):
        e=KerasEngine(Path('.'),allow_legacy=True)
        with patch.dict('sys.modules',{'tensorflow':None}):
            code,result=handle_prediction(self.png(),e)
        self.assertEqual(code,503);self.assertNotIn('probabilities',result)
    def test_zero_heatmap_is_no_highlight(self):
        im=Image.new('RGB',(40,30),'red')
        np.testing.assert_array_equal(np.asarray(overlay_heatmap(im,np.zeros((7,7)))),np.asarray(im))
        with self.assertRaises(ValueError):overlay_heatmap(im,np.full((7,7),float('nan')))
    def test_http_routes_without_network(self):
        # Exercise real request parsing/handlers through in-memory streams; no browser/network claim.
        class Connection:
            def __init__(self,raw):self.source=io.BytesIO(raw);self.output=io.BytesIO()
            def makefile(self,*args,**kwargs):return self.source
            def sendall(self,data):self.output.write(data)
            def settimeout(self,t):pass
        e=KerasEngine(Path('.'));handler=handler_factory(e,Path('web'))
        for request,expected in [(b'GET / HTTP/1.0\r\n\r\n',b'200'),(b'GET /api/status HTTP/1.0\r\n\r\n',b'200'),(b'GET /../../etc/passwd HTTP/1.0\r\n\r\n',b'404')]:
            conn=Connection(request);handler(conn,('127.0.0.1',1),None)
            self.assertIn(expected,conn.output.getvalue().splitlines()[0])
        raw=b'POST /api/predict HTTP/1.0\r\nContent-Length: 3\r\n\r\nBAD'
        conn=Connection(raw);handler(conn,('127.0.0.1',1),None)
        self.assertIn(b'400',conn.output.getvalue().splitlines()[0])

if __name__=='__main__':unittest.main()
