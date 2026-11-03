import unittest
from gesture_cv.training import validate_run_branch

class MethodBranch(unittest.TestCase):
    def test_pretraining_ablation_roles_and_guardrails(self):
        contract={'method':'convnext_pretraining'}
        real=[{'origin':'real'}]
        for role,weights in [('S1','imagenet'),('A0',None)]:
            config={'run_role':role,'architecture':'convnext_tiny','backbone_weights':weights,'fine_tune_epochs':0}
            validate_run_branch(config,contract,real)
            for wrong in [dict(config,backbone_weights='wrong'),dict(config,architecture='mobilenetv2'),dict(config,fine_tune_epochs=1)]:
                with self.assertRaises(ValueError):validate_run_branch(wrong,contract,real)
            with self.assertRaisesRegex(ValueError,'real-only'):
                validate_run_branch(config,contract,[{'origin':'synthetic'}])
    def test_old_generator_guard_is_not_silently_removed(self):
        with self.assertRaisesRegex(ValueError,'synthetic'):
            validate_run_branch({'run_role':'S1','architecture':'mobilenetv2'},{},[{'origin':'real'}])
        with self.assertRaises(ValueError):
            validate_run_branch({'run_role':'A0','architecture':'convnext_tiny'},{},[{'origin':'real'}])
if __name__=='__main__':unittest.main()
