import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import cv2, time
from core.preprocess import standardize
from core.agent import inspect_standardized

print("--- SMOKE TESTS ---")
# 1. Normal board
p_norm = Path("datasets/VisA_pytorch/1cls/pcb1/test/good/0011.JPG")
t0 = time.time()
res_norm = inspect_standardized(standardize(cv2.imread(str(p_norm))).image)
print(f"1. Normal: {res_norm['decision']} ({res_norm['decision_path']}) in {(time.time()-t0)*1000:.1f}ms")

# 2. Defective board
p_bad = Path("datasets/VisA_pytorch/1cls/pcb1/test/bad/000.JPG")
t0 = time.time()
res_bad = inspect_standardized(standardize(cv2.imread(str(p_bad))).image)
print(f"2. Defect: {res_bad['decision']} ({res_bad['decision_path']}) in {(time.time()-t0)*1000:.1f}ms")

# 3. Blurred image
img_blur = cv2.GaussianBlur(cv2.imread(str(p_norm)), (25, 25), 10.0)
t0 = time.time()
res_blur = inspect_standardized(standardize(img_blur).image)
print(f"3. Blur: {res_blur['decision']} ({res_blur['decision_path']}) in {(time.time()-t0)*1000:.1f}ms")

# 4. Glare / Poor lighting
img_glare = cv2.imread(str(p_norm)).copy()
img_glare[200:400, 200:400] = 255
t0 = time.time()
res_glare = inspect_standardized(standardize(img_glare).image)
print(f"4. Glare: {res_glare['decision']} ({res_glare['decision_path']}) in {(time.time()-t0)*1000:.1f}ms")

# 5. Alignment failure (rotated 90 degrees)
img_rot = cv2.rotate(cv2.imread(str(p_norm)), cv2.ROTATE_90_CLOCKWISE)
t0 = time.time()
res_rot = inspect_standardized(standardize(img_rot).image)
print(f"5. Rotated (Align Fail): {res_rot['decision']} ({res_rot['decision_path']}) in {(time.time()-t0)*1000:.1f}ms")
