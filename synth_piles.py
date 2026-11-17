"""
จำลองภาพ "เมล็ดวางกอง" จากภาพจัดเรียง (copy-paste) — ใช้ตอนเทรนเท่านั้น
เพื่อให้โมเดลเห็น crop แบบเมล็ดชิด/ซ้อนกันเหมือนภาพ test

    python synth_piles.py <layout> <out_dir> [n_per_image]   # สร้างภาพกองไว้ดู/ใช้เป็น val
"""
import os
import sys
import glob

import cv2
import numpy as np

from infer import read_image_bgr, foreground_mask


def _cut_beans(img):
    """ตัดเมล็ดเดี่ยวพร้อม mask จากภาพจัดเรียง"""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    fg_small = foreground_mask(small)
    fg = cv2.resize(fg_small, (w, h), interpolation=cv2.INTER_NEAREST)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    beans = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if 8000 < a < 400000:
            m = cv2.erode((lab[y:y + bh, x:x + bw] == i).astype(np.uint8), np.ones((9, 9), np.uint8))
            beans.append((img[y:y + bh, x:x + bw].copy(), m))
    return beans, fg


def _clean_background(img, fg):
    """ลบเมล็ดเดิมออกจากภาพ (inpaint ที่ความละเอียดต่ำ) ให้เหลือแต่กระดาษ"""
    h, w = img.shape[:2]
    hole = cv2.dilate(fg, np.ones((41, 41), np.uint8))
    q = 4
    inp = cv2.inpaint(cv2.resize(img, (w // q, h // q)),
                      cv2.resize(hole, (w // q, h // q), interpolation=cv2.INTER_NEAREST), 5, cv2.INPAINT_TELEA)
    inp = cv2.GaussianBlur(cv2.resize(inp, (w, h)), (31, 31), 0)
    bg = img.copy()
    bg[hole > 0] = inp[hole > 0]
    return bg


def make_piles(img, rng, n_piles=1):
    beans, fg = _cut_beans(img)
    if not beans:
        return []
    base_bg = _clean_background(img, fg)
    h, w = img.shape[:2]
    bean_d = np.median([max(b.shape[:2]) for b, _ in beans])
    out = []
    for _ in range(n_piles):
        bg = base_bg.copy()
        cx = w // 2 + int(rng.uniform(-0.15, 0.15) * w)
        cy = h // 2 + int(rng.uniform(-0.15, 0.15) * h)
        centers = []
        order = rng.permutation(np.resize(np.arange(len(beans)), int(rng.integers(12, 20))))
        for j in order:
            b, m = beans[j]
            gap = rng.uniform(0.55, 0.8) * bean_d  # ระยะศูนย์กลาง < ขนาดเมล็ด = ชิด/ซ้อนบางส่วน
            p = np.array([cx, cy], float)
            for _try in range(500):
                if not centers:
                    break
                t = rng.uniform(0, 2 * np.pi)
                p = centers[rng.integers(len(centers))] + gap * np.array([np.cos(t), np.sin(t)])
                if min(np.hypot(*(p - c)) for c in centers) >= gap * 0.98:
                    break
            M = cv2.getRotationMatrix2D((b.shape[1] / 2, b.shape[0] / 2), rng.uniform(0, 360), 1.0)
            size = int(np.hypot(*b.shape[:2])) + 2
            M[0, 2] += (size - b.shape[1]) / 2
            M[1, 2] += (size - b.shape[0]) / 2
            br = cv2.warpAffine(b, M, (size, size))
            mr = cv2.warpAffine(m, M, (size, size), flags=cv2.INTER_NEAREST)
            px, py = int(p[0] - size / 2), int(p[1] - size / 2)
            if px < 0 or py < 0 or px + size + 6 > w or py + size + 8 > h:
                continue
            centers.append(p)
            sh = cv2.GaussianBlur(mr.astype(np.float32), (31, 31), 0)[..., None]  # เงาจางๆ
            roi = bg[py + 8:py + 8 + size, px + 6:px + 6 + size].astype(np.float32)
            bg[py + 8:py + 8 + size, px + 6:px + 6 + size] = (roi * (1 - 0.25 * sh)).astype(np.uint8)
            roi = bg[py:py + size, px:px + size]
            roi[mr > 0] = br[mr > 0]
        out.append(bg)
    return out


if __name__ == '__main__':
    layout, out_dir = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = np.random.default_rng(123)
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join('dataset', '*', f'{layout}_*.jpg'))):
        cls = os.path.basename(os.path.dirname(path))
        for k, pile in enumerate(make_piles(read_image_bgr(path), rng, n)):
            cv2.imwrite(os.path.join(out_dir, f'pile{k}_{layout}_{cls}.jpg'), pile, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(path, '->', n, 'piles')
