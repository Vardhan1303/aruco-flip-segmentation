import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
TARGET_CLASS_ID = 39  # bottle
CAMERA_INDEXES = [1, 0, 2, 3, 4]
YOLO_IMGSZ = 640  # yolov8n's native/optimal input size; must be a multiple of 32


def resize_for_yolo(frame, target=YOLO_IMGSZ):
    """Downscale so the longer side matches YOLO's native input size, rounded
    to a multiple of 32, before inference. Ultralytics does internally
    letterbox to `imgsz` regardless of input size, but a raw 3072x4096 phone
    photo is very far from what the model was tuned/tested at (and from what
    the webcam already captures at) — resizing explicitly keeps photo mode
    behaving the same way live mode already does.
    """
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_w = max(32, int(round(w * scale / 32)) * 32)
    new_h = max(32, int(round(h * scale / 32)) * 32)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def annotate(frame, model, conf):
    results = model(frame, classes=[TARGET_CLASS_ID], conf=conf, imgsz=YOLO_IMGSZ, verbose=False)
    annotated = results[0].plot()

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        cv2.putText(
            annotated, f"Detections: {len(boxes)}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
    else:
        cv2.putText(
            annotated, "NOT DETECTED", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        )
    return annotated


def run_photo(model, image_path: str, conf: float):
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    frame = resize_for_yolo(frame)
    annotated = annotate(frame, model, conf)

    out_path = Path(image_path).stem + "_yolo.png"
    cv2.imwrite(out_path, annotated)
    print(f"Saved {out_path}")

    cv2.imshow("Bottle Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_camera(indexes):
    for idx in indexes:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Connected to camera index {idx}")
            return cap
        cap.release()
    return None


def run_live(model, conf: float, cam_index: int):
    # try the requested index first, then fall back through the rest
    indexes = [cam_index] + [i for i in CAMERA_INDEXES if i != cam_index]
    cap = open_camera(indexes)
    if cap is None:
        raise RuntimeError("No webcam was found. Plug in your external camera and try again.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # No manual resize here — live capture is already close to YOLO's
        # native scale (typical webcam frames are 640-1280px), same as the
        # working webcam script. imgsz still lets ultralytics letterbox
        # correctly regardless.
        annotated = annotate(frame, model, conf)
        cv2.imshow("Bottle Detection (live)", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["photo", "live"], required=True,
                     help="photo: run once on a single image file. live: open a webcam feed.")
    ap.add_argument("--image", help="path to a single photo (required for --mode photo, any resolution)")
    ap.add_argument("--cam-index", type=int, default=1, help="preferred camera index (--mode live)")
    ap.add_argument("--conf", type=float, default=0.45)
    args = ap.parse_args()

    model = YOLO(MODEL_PATH)

    if args.mode == "photo":
        if not args.image:
            raise SystemExit("--image is required when --mode photo")
        run_photo(model, args.image, args.conf)
    else:
        run_live(model, args.conf, args.cam_index)


if __name__ == "__main__":
    main()
