import argparse
from pathlib import Path

import cv2
import numpy as np

# Reuses segment.py's classes/functions directly rather than duplicating them
# (importing it only defines things — main() is guarded — so this doesn't
# open a webcam by itself).
from segment import FlipSegmenter, make_detector, marker_center_and_size, ARUCO_DICTS

CAMERA_INDEXES = [1, 0, 2, 3, 4]
MASK_COLOR = (0, 255, 0)
CENTER_COLOR = (0, 0, 255)


def segment_frame(frame, flip: "FlipSegmenter", detector, args):
    """Runs ArUco detection + FLIP segmentation on one frame (photo or live),
    returns the annotated frame. Same core logic either way — only the
    source of `frame` differs between modes."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    corners_list, ids, _rejected = detector.detectMarkers(frame)
    if ids is None:
        cv2.putText(overlay, "NOT DETECTED", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return overlay

    ids_flat = np.asarray(ids).reshape(-1)
    count = 0

    for i, marker_corners in enumerate(corners_list):
        marker_id = int(ids_flat[i])
        try:
            pts = marker_corners.reshape(4, 2)
            center, side_px = marker_center_and_size(pts)
            cx, cy = center

            if args.full_frame:
                x0, y0, x1, y1 = 0, 0, w, h
            else:
                half_w = max(side_px * args.roi_scale_x / 2.0, 20)
                half_h = max(side_px * args.roi_scale_y / 2.0, 20)
                cy_shifted = cy + args.roi_offset_y * half_h
                x0 = int(np.clip(cx - half_w, 0, w - 1))
                x1 = int(np.clip(cx + half_w, 0, w))
                y0 = int(np.clip(cy_shifted - half_h, 0, h - 1))
                y1 = int(np.clip(cy_shifted + half_h, 0, h))
                if x1 - x0 < 10 or y1 - y0 < 10:
                    continue

            roi_bgr = frame[y0:y1, x0:x1]
            crop_h, crop_w = roi_bgr.shape[:2]
            if crop_h >= crop_w:
                target_h = args.roi_size
                target_w = max(8, round(crop_w * (args.roi_size / crop_h)))
            else:
                target_w = args.roi_size
                target_h = max(8, round(crop_h * (args.roi_size / crop_w)))
            roi_resized = cv2.resize(roi_bgr, (target_w, target_h))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

            rel_x = (cx - x0) / (x1 - x0) * 2 - 1
            rel_y = (cy - y0) / (y1 - y0) * 2 - 1

            mask = flip.segment(roi_rgb, rel_x, rel_y, args.sigma_x, args.sigma_y)
            alpha_full = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            alpha = np.clip(alpha_full * args.mask_alpha, 0, 1)[..., None].astype(np.float32)
            region = overlay[y0:y1, x0:x1].astype(np.float32)
            colored = np.full_like(region, MASK_COLOR, dtype=np.float32)
            blended = region * (1 - alpha) + colored * alpha
            overlay[y0:y1, x0:x1] = blended.astype(np.uint8)

            cv2.circle(overlay, (int(cx), int(cy)), 5, CENTER_COLOR, -1)
            count += 1
        except Exception as e:
            print(f"[WARN] segmentation failed for marker {marker_id}: {e}")

    label = f"Detections: {count}" if count else "NOT DETECTED"
    color = (0, 255, 0) if count else (0, 0, 255)
    cv2.putText(overlay, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return overlay


def resize_for_flip(frame, target):
    """Downscale so the longer side is `target` pixels. Same reasoning as
    baseline_yolo.py's resize step: cv2.aruco's detector uses default
    adaptive-threshold window sizes tuned for typical camera-frame
    resolutions (the live webcam feed). A phone photo's marker can be
    physically huge in pixel terms on an unresized 3072x4096 image, which is
    a likely reason detection fails on photos but not live video even though
    the marker looks perfectly readable to the eye."""
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def run_photo(flip, detector, args):
    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    frame = resize_for_flip(frame, args.max_size)
    annotated = segment_frame(frame, flip, detector, args)

    out_path = Path(args.image).stem + "_flip.png"
    cv2.imwrite(out_path, annotated)
    print(f"Saved {out_path}")

    cv2.imshow("ArUco + FLIP Segmentation", annotated)
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


def run_live(flip, detector, args):
    indexes = [args.cam_index] + [i for i in CAMERA_INDEXES if i != args.cam_index]
    cap = open_camera(indexes)
    if cap is None:
        raise RuntimeError("No webcam was found. Plug in your external camera and try again.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        annotated = segment_frame(frame, flip, detector, args)
        cv2.imshow("ArUco + FLIP Segmentation (live)", annotated)

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
    ap.add_argument("--max-size", type=int, default=1280,
                     help="--mode photo only: downscale so the longer side is at most this many "
                          "pixels before ArUco detection (phone photos are often 3000px+)")
    ap.add_argument("--model", choices=["tiny", "small", "middle"], default="small")
    ap.add_argument("--dict", choices=list(ARUCO_DICTS.keys()), default="DICT_6X6_250")
    ap.add_argument("--roi-scale-x", type=float, default=6.0)
    ap.add_argument("--roi-scale-y", type=float, default=11.0)
    ap.add_argument("--roi-offset-y", type=float, default=0.35)
    ap.add_argument("--full-frame", action="store_true")
    ap.add_argument("--roi-size", type=int, default=256)
    ap.add_argument("--sigma-x", type=float, default=0.28)
    ap.add_argument("--sigma-y", type=float, default=0.42)
    ap.add_argument("--num-tokens", type=int, default=512)
    ap.add_argument("--mask-alpha", type=float, default=0.6)
    args = ap.parse_args()

    detector = make_detector(args.dict)
    flip = FlipSegmenter(args.model, num_tokens=args.num_tokens)

    if args.mode == "photo":
        if not args.image:
            raise SystemExit("--image is required when --mode photo")
        run_photo(flip, detector, args)
    else:
        run_live(flip, detector, args)


if __name__ == "__main__":
    main()
