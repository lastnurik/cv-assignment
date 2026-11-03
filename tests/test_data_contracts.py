import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from gesture_cv import CLASS_NAMES
from gesture_cv.collection import start_session,save_clean_image
from gesture_cv.preprocessing import open_rgb,camera_rgb,crop_rgb,model_input,probabilities
from gesture_cv.dataset import sha256,validate_manifest
from gesture_cv.metrics import classification_metrics,compare_aligned

class DataContracts(unittest.TestCase):
    def test_collection_two_sessions_clean_and_no_overwrite(self):
        with TemporaryDirectory() as td:
            image=Image.new('RGB',(640,480),(20,40,80))
            session=start_session(td,environment='daylight')
            first=save_clean_image(session,'peace',image);digest=sha256(first)
            display=image.copy();ImageDraw.Draw(display).text((10,10),'peace label',fill='white')
            second=save_clean_image(session,'peace',image)
            other=start_session(td,environment='daylight')
            third=save_clean_image(other,'peace',image)
            self.assertEqual(len({first,second,third}),3)
            self.assertEqual(sha256(first),digest)
            for path in [first,second,third]:np.testing.assert_array_equal(np.asarray(open_rgb(path)),np.asarray(image))
            with patch('gesture_cv.collection.uuid4') as uuid:
                uuid.return_value.hex=first.stem
                with self.assertRaises(FileExistsError):save_clean_image(session,'peace',display)
            self.assertEqual(sha256(first),digest)
    def test_camera_rgb_parity_and_no_double_crop(self):
        rng=np.random.default_rng(42);rgb=rng.integers(0,256,(480,640,3),dtype=np.uint8)
        image=Image.fromarray(rgb)
        camera=camera_rgb(rgb[...,::-1],mirror=False)
        np.testing.assert_array_equal(model_input(image),model_input(camera))
        cropped,box=crop_rgb(image)
        self.assertEqual(box,(0,96,640,480))
        self.assertEqual(cropped.size,(640,384))
        np.testing.assert_array_equal(model_input(image),model_input(cropped,already_cropped=True))
        self.assertFalse(np.array_equal(model_input(image),model_input(cropped)))
    def test_camera_mirror_and_color_fixture(self):
        bgr=np.array([[[255,0,0],[0,0,255]]],dtype=np.uint8)
        np.testing.assert_array_equal(np.asarray(camera_rgb(bgr,mirror=False)),[[[0,0,255],[255,0,0]]])
        np.testing.assert_array_equal(np.asarray(camera_rgb(bgr,mirror=True)),[[[255,0,0],[0,0,255]]])
    def test_exif_applied_once(self):
        with TemporaryDirectory() as td:
            im=Image.new('RGB',(30,20),'red');exif=im.getexif();exif[274]=6
            p=Path(td)/'rot.jpg';im.save(p,exif=exif)
            loaded=open_rgb(p);self.assertEqual(loaded.size,(20,30))
            self.assertNotIn(274,loaded.getexif())
    def test_confidence_not_resoftmaxed(self):
        p=probabilities([[.99,.005,.005]])
        self.assertEqual(float(p.max()),.99)
        for bad in ([1,1,1],[float('nan'),0,0],[-.1,.1,1],[.2,.3],[[.1,.2,.7],[.2,.2,.6]]):
            with self.assertRaises(ValueError):probabilities(bad)
    def fixture_manifest(self,root):
        rows=[]
        for si,split in enumerate(['train','val','test']):
            for li,label in enumerate(CLASS_NAMES):
                p=root/f'{split}_{label}.png';Image.new('RGB',(8,8),(si*60,li*60,30)).save(p)
                rows.append({'sample_id':p.stem,'path':p.name,'sha256':sha256(p),'source_sha256':sha256(p),
                    'label':label,'origin':'real','group_id':split,'split':split,'environment':'fixture',
                    'consent_status':'not_applicable','review_status':'accepted'})
        return {'class_names':list(CLASS_NAMES),'rows':rows}
    def test_ready_dataset_and_negative_cases(self):
        with TemporaryDirectory() as td:
            root=Path(td);m=self.fixture_manifest(root)
            self.assertTrue(validate_manifest(m,root)['passed'])
            for field,value in [('split','unassigned'),('review_status','pending'),('consent_status','unknown'),('group_id','train'),('origin','synthetic')]:
                changed=copy.deepcopy(m);changed['rows'][-1][field]=value
                self.assertFalse(validate_manifest(changed,root)['passed'],(field,value))
            changed=copy.deepcopy(m);changed['rows'][-1]['source_sha256']=m['rows'][0]['source_sha256']
            self.assertFalse(validate_manifest(changed,root)['passed'])
            (root/m['rows'][0]['path']).write_bytes(b'broken')
            self.assertFalse(validate_manifest(m,root)['passed'])
    def test_metrics_known_confusion_matrix(self):
        # historical matrix used only as an arithmetic fixture, not a new model run
        cm=[[30,0,0],[14,5,8],[0,0,35]];rows=[]
        for i,row in enumerate(cm):
            for j,count in enumerate(row):
                for k in range(count):rows.append({'sample_id':f'{i}-{j}-{k}','label':CLASS_NAMES[i],'prediction':CLASS_NAMES[j],'environment':'fixture'})
        r=classification_metrics(rows)
        self.assertEqual(r['confusion_matrix'],cm);self.assertAlmostEqual(r['accuracy'],70/92)
        self.assertAlmostEqual(r['per_class']['stop']['recall'],5/27)
        self.assertAlmostEqual(r['per_class']['peace']['precision'],30/44)
        compare_aligned({'a':rows,'b':list(reversed(rows))})
        with self.assertRaises(ValueError):compare_aligned({'a':rows,'b':rows[:-1]})
        with self.assertRaises(ValueError):classification_metrics(rows+[rows[0]])

if __name__=='__main__':unittest.main()
