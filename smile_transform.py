import cv2
import numpy as np

def _normalize_mask(mask, target_shape_hw):
    """
    Ensures mask becomes uint8 HxW in {0,255}, matching target H,W.
    Accepts masks in shapes: HxW, HxWx1, 1x1xHxW, 1xHxW, HxWx3, etc.
    """
    m = np.asarray(mask)

    # Remove extra dims like (1,1,H,W) or (1,H,W)
    m = np.squeeze(m)

    # If mask became 3-channel, convert to gray
    if m.ndim == 3:
        # HxWxC -> take first channel (or convert properly)
        if m.shape[2] == 3:
            m = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            m = m[:, :, 0]

    # Now must be 2D
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got shape={m.shape}")

    # Resize if needed
    H, W = target_shape_hw
    if m.shape[0] != H or m.shape[1] != W:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    # Convert range to 0..255
    if m.dtype != np.uint8:
        m = m.astype(np.float32)
        # If mask looks like 0..1, scale up
        if m.max() <= 1.0:
            m = m * 255.0
        m = np.clip(m, 0, 255).astype(np.uint8)

    # Ensure binary-ish
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def smile_whitening_transform(frame_bgr, teeth_mask_255,
                              brighten_L=28, reduce_yellow=10,
                              feather_ksize=31):
    """
    frame_bgr: HxWx3 BGR image
    teeth_mask_255: mask that can be HxW / HxWx1 / 1x1xHxW etc.
    returns: transformed BGR image
    """
    out = frame_bgr.copy()
    H, W = out.shape[:2]

    # ✅ Fix mask shape here
    mask = _normalize_mask(teeth_mask_255, (H, W))

    # Feather edges for smooth blending
    k = int(feather_ksize)
    if k % 2 == 0:
        k += 1
    mask_blur = cv2.GaussianBlur(mask, (k, k), 0)

    alpha = (mask_blur.astype(np.float32) / 255.0)[:, :, None]  # HxWx1

    # LAB transform for natural whitening
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    L2 = np.clip(L + brighten_L, 0, 255)
    B2 = np.clip(B - reduce_yellow, 0, 255)

    lab2 = cv2.merge([L2, A, B2]).astype(np.uint8)
    whitened = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    smooth = cv2.bilateralFilter(whitened, d=7, sigmaColor=40, sigmaSpace=40)

    # ✅ Shapes now match: alpha (H,W,1) with images (H,W,3)
    result = (alpha * smooth + (1.0 - alpha) * out).astype(np.uint8)
    return result
