# -*- coding: utf-8 -*-
"""
画像の異常検知
sparse_modeling_defect-detection.py

This is a copy modify from defect-detection.ipynb
from https://github.com/hacarus/codezine-sparse-modeling/tree/master/5.


# 
"""

#!pip install japanize-matplotlib
#!pip install spm-image  # use spm-image  https://github.com/hacarus/spm-image

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import time

# %matplotlib inline
# %load_ext autoreload
# %autoreload 1

plt.rcParams["font.family"] = "MS Gothic"  # use for Windows  #"IPAexGothic"

#import warnings
#warnings.filterwarnings('ignore')

"""## 正常画像から辞書を作成"""

#!pwd
#!mkdir img
#!mkdir img/result

# 正常画像を読み込み
ok1="img/Figure_1b.png" # "img/wood-ok2.jpg"
ng1="img/Figure_2b.png" # "img/wood-ng2.jpg"
#ok_img = np.asarray(Image.open("img/wood-ok.jpg").convert('L'))
ok_img = np.asarray(Image.open(ok1).convert('L')) 
plt.imshow(ok_img, cmap='gray')
plt.title('正常画像')

# 正常画像は再構成できるようにしたいが、それ以外は再構成できないように、表現力を小さく設定する
#patch_size = (16, 16)
patch_size = (16, 4)
n_components = 10
transform_n_nonzero_coefs = 3
max_iter = 30

from sklearn.preprocessing import StandardScaler
from spmimage.feature_extraction.image import extract_simple_patches_2d, reconstruct_from_simple_patches_2d

# 学習用データの用意
scl = StandardScaler()
patches = extract_simple_patches_2d(ok_img, patch_size)
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
Y = scl.fit_transform(patches)

# Commented out IPython magic to ensure Python compatibility.
# %%time
s_time = time.time()
from spmimage.decomposition import KSVD
# 
# # 辞書学習
ksvd = KSVD(n_components=n_components, transform_n_nonzero_coefs=transform_n_nonzero_coefs, max_iter=max_iter)
X = ksvd.fit_transform(Y)
D = ksvd.components_
end = time.time() - s_time
print(f'{end}秒掛かりました')

# 正常画像が再構成できるかどうか確かめる
reconstructed_patches = np.dot(X, D)
reconstructed_patches = scl.inverse_transform(reconstructed_patches)
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size[0], patch_size[1])
reconstructed_img = reconstruct_from_simple_patches_2d(reconstructed_patches, ok_img.shape)
reconstructed_img[reconstructed_img < 0] = 0
reconstructed_img[reconstructed_img > 255] = 255
reconstructed_img = reconstructed_img.astype(np.uint8)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(ok_img, cmap='gray')
plt.title("正常画像")
plt.subplot(1,2,2)
plt.imshow(reconstructed_img, cmap='gray')
plt.title("再構成画像")
plt.savefig("./img/result/ok-reconstruct.png")
plt.clf()
"""## 異常画像を再構成
正常画像で学習された辞書を使って、異常画像に対して再構成し、そのときの再構成誤差を確認する。
"""

# 異常画像を読み込み

#ng_img = np.asarray(Image.open("img/wood-ng.jpg").convert('L'))
ng_img = np.asarray(Image.open(ng1).convert('L'))
plt.imshow(ng_img, cmap='gray')
plt.title('異常画像')

# 異常画像に対するスパースコードを求める
patches = extract_simple_patches_2d(ng_img, patch_size)
patches = patches.reshape(-1, np.prod(patch_size)).astype(np.float64)
Y = scl.transform(patches)
X = ksvd.transform(Y)

# 再構成画像を計算
reconstructed_patches = np.dot(X, D)
reconstructed_patches = scl.inverse_transform(reconstructed_patches)
reconstructed_patches = reconstructed_patches.reshape(-1, patch_size[0], patch_size[1])
reconstructed_img = reconstruct_from_simple_patches_2d(reconstructed_patches, ng_img.shape)
reconstructed_img[reconstructed_img < 0] = 0
reconstructed_img[reconstructed_img > 255] = 255
reconstructed_img = reconstructed_img.astype(np.uint8)

plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.imshow(ng_img, cmap='gray')
plt.title("異常画像")
plt.subplot(1,2,2)
plt.imshow(reconstructed_img, cmap='gray')
plt.title("再構成画像")
plt.savefig("./img/result/ng-reconstruct.png")
plt.clf()

# 再構成誤差を計算し、ヒストグラムを描画してみる
diff_img = np.abs(np.array(ng_img, dtype=np.float) - np.array(reconstructed_img, dtype=np.float))
sns.distplot(diff_img.ravel())
plt.title("Histogram of Absolute Error")
plt.savefig("./img/result/error-hist.png")
plt.clf()
# 誤差が小さい部分と大きい部分で二値化を行う。
bin_diff_img = np.zeros_like(diff_img)
bin_diff_img[diff_img >= 10] = 255

# 元画像、再構成画像、diff画像を並べて描画
plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.imshow(ng_img, cmap='gray')
plt.title("異常画像")
plt.subplot(1,3,2)
plt.imshow(reconstructed_img, cmap='gray')
plt.title("再構成画像")
plt.subplot(1,3,3)
plt.imshow(bin_diff_img, cmap='gray')
plt.title("差分画像")
plt.savefig("./img/result/detected-result.png")