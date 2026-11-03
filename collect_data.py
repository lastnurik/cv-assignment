"""Capture clean gesture photos without overwrites. Run --help for session metadata."""
import argparse
from pathlib import Path
from gesture_cv import CLASS_NAMES
from gesture_cv.collection import start_session, save_clean_image
from gesture_cv.preprocessing import camera_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw_sessions'))
    parser.add_argument('--environment', required=True, help='e.g. room_daylight or room_dim')
    parser.add_argument('--participant', default='unknown', help='Pseudonymous identifier')
    parser.add_argument('--consent', choices=['unknown','private_use_confirmed','public_release_confirmed'], default='unknown')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit('OpenCV is missing. Install the runtime requirements first.') from exc
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        raise SystemExit('Cannot open camera; no session created.')
    folder = start_session(args.root, environment=args.environment, participant=args.participant, consent=args.consent)
    print('Session:', folder, '\nSPACE saves; Q moves to next class; ESC ends.')
    try:
        for label in CLASS_NAMES:
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera stopped returning frames.')
                # Snapshot clean pixels before writing anything on the display.
                clean_rgb = camera_rgb(frame, mirror=True)
                display = cv2.flip(frame, 1).copy()
                cv2.putText(display, f'{label}: {count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                cv2.putText(display, 'SPACE save | Q next | ESC quit', (10,80),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                cv2.imshow('Collect gestures', display)
                key = cv2.waitKey(1) & 0xff
                if key == 27:
                    return
                if key in (ord('q'), ord('Q')):
                    break
                if key == ord(' '):
                    path = save_clean_image(folder, label, clean_rgb)
                    count += 1
                    print(path)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
