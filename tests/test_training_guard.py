import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from gesture_cv.training import preflight
from gesture_cv.dataset import sha256

class TrainingGuard(unittest.TestCase):
    def test_draft_contract_rejected_before_tensorflow(self):
        with self.assertRaisesRegex(ValueError,'not frozen'):
            preflight('configs/run.example.json',Path('.'))
    def test_modified_contract_and_incomplete_settings_rejected(self):
        with TemporaryDirectory() as td:
            root=Path(td);contract=root/'contract.json'
            contract.write_text(json.dumps({'status':'FROZEN'}))
            config={'contract':'contract.json','contract_sha256':'wrong','run_role':'B1'}
            cp=root/'config.json';cp.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'Contract changed'):preflight(cp,root)
            config['contract_sha256']=sha256(contract);cp.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'do not match'):preflight(cp,root)
if __name__=='__main__':unittest.main()
