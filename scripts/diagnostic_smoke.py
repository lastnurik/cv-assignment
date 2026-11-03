"""Historical A1 integration checks, never final A2 evaluation or accuracy."""
import json
from pathlib import Path
import numpy as np
from gesture_cv import CLASS_NAMES
from gesture_cv.dataset import sha256, write_json
from gesture_cv.inference import KerasEngine
from gesture_cv.preprocessing import open_rgb, camera_rgb, model_input
from gesture_cv.webapp import handle_prediction


def main():
    engine = KerasEngine('.', allow_legacy=True)
    engine.load()
    records = []
    maps = []
    for label in CLASS_NAMES:
        path = sorted(Path('dataset', label).glob('*.jpg'))[0]
        im = open_rgb(path)
        x = model_input(im, config=engine.config)
        camera = camera_rgb(np.asarray(im)[..., ::-1], mirror=False)
        np.testing.assert_array_equal(x, model_input(camera, config=engine.config))
        expected = engine.model(x, training=False).numpy()[0]
        code, result = handle_prediction(path.read_bytes(), engine)
        assert code == 200, result
        np.testing.assert_allclose(list(result['probabilities'].values()), expected, atol=1e-6)
        heatmaps = [engine.explain(x, target)[0] for target in range(3)]
        assert all(np.isfinite(h).all() for h in heatmaps)
        assert any(np.any(h > 0) for h in heatmaps)
        assert any(not np.allclose(heatmaps[0], h) for h in heatmaps[1:])
        maps.append(heatmaps[0])
        records.append({'sample': str(path), 'sha256': sha256(path),
                        'label': label, 'prediction': result['prediction'],
                        'probabilities': result['probabilities'],
                        'inference_ms': result['inference_ms'],
                        'explanation_ms': result['explanation_ms'],
                        'target_maps_differ': True,
                        'predicted_target_nonzero': result['explanation_valid']})
    assert any(not np.allclose(maps[0], h) for h in maps[1:])
    loaded_model = engine.model
    for _ in range(3):
        code, result = handle_prediction(path.read_bytes(), engine)
        assert code == 200
        np.testing.assert_allclose(list(result['probabilities'].values()), expected, atol=1e-6)
        assert engine.model is loaded_model
    write_json('artifacts/qa/diagnostic_smoke.json', {
        'purpose': 'Historical leakage-affected A1 diagnostic only; no A2 gate or accuracy claim',
        'passed': True, 'model_sha256': sha256(engine.path), 'class_names': list(CLASS_NAMES),
        'preprocessing': engine.config.to_dict(), 'records': records,
        'input_and_target_maps_differ': True, 'repeat_requests_stable': True,
        'file_camera_model_input_parity': True, 'handler_direct_probability_parity': True,
        'status': engine.status()})

if __name__ == '__main__':
    main()
