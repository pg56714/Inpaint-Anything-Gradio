# 安全性紀錄

[English](security-log.md) | 繁體中文

- 日期: 2026-10-25
- 分支: `fix/security-upgrades`
- 範圍: `requirements.txt`、`lama/`

## 摘要

推論流程不再載入或安裝沒有修正版的 imgaug，kornia 解除固定版本以相容目前的 torch，gradio 改為要求已修補的版本。pillow 因為 gradio 限制版本，仍有未修正的漏洞。

## 修改內容

| 項目 | 修改前 | 修改後 | 原因 |
|---|---|---|---|
| gradio | unpinned | >=6.16 | 任意檔案讀取、SSRF、路徑穿越漏洞（6.16 修正） |
| albumentations | ==0.5.2 | removed | 會帶入 imgaug（不安全的反序列化，沒有修正版） |
| kornia | ==0.5.0 | unpinned | 0.5.0 不支援目前的 torch |
| `lama/saicinpainting/training/data/datasets.py` | - | lazy import | albumentations / imgaug 改在 `get_transforms()` 內才載入，只有訓練會用到 |
| `lama/.../countless/requirements.txt` | Pillow>=6.2.0 | Pillow>=10.3.0 | 舊版 Pillow 有已知漏洞 |

## 使用前須知

- 訓練 LaMa 需要另外手動安裝 albumentations 0.5.2（因此也會裝 imgaug）；推論不再載入它們。
- `lama/requirements.txt` 與 `lama/docker/` 是上游的訓練用檔案，沒有修改，仍固定舊版本。

## 尚未解決的問題

### 被上游限制的套件版本

| 套件 | 版本 | 漏洞 | 無法修正的原因 | 影響與因應 |
|---|---|---|---|---|
| pillow | 12.3.0 | 41 筆（13.0.0 修正） | gradio 6.16 以上要求 `pillow<13`，目前沒有允許 Pillow 13 的 gradio 版本。 | 解碼惡意製作的圖片（例如 TIFF、PDF）可能讓程式當掉或越界讀取記憶體。請只在本機執行 Gradio 介面（不要開 `share=True`、不要對外開放連接埠），只開啟可信任的圖片；等 gradio 允許 Pillow 13 後再升級。 |

## 驗證方式

- 以 osv-scanner 掃描解析後的 `requirements.txt`，只剩 pillow。
- torch 2.14.1（CPU）、kornia 0.9.0、pytorch-lightning 2.6.6：`lama_inpaint` 不需 albumentations 即可載入，kornia 的 refinement 運算可正常執行。
- 把 SAM、LaMa、Stable Diffusion 模型換成假模組後，Gradio 介面能正常建立並提供服務。

## 未驗證項目

- 用 GPU 與實際權重執行推論（測試環境沒有 GPU 與權重檔）。
- LaMa 訓練流程。
