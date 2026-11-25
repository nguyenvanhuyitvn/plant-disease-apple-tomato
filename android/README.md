# Ứng dụng Android — Bệnh lá táo & cà chua

Chụp hoặc chọn ảnh một lá → mô hình MobileNetV3-Large (cấu hình đề xuất, INT8, khoảng 4,7 MB mỗi loài) chạy
**offline trên máy** bằng ONNX Runtime → tên bệnh (tiếng Việt), độ tin cậy đã hiệu chỉnh và ba khả năng cao nhất.

- Chọn loài: **Tự động** (chạy cả hai mô hình, chọn loài có năng lượng vượt ngưỡng τ_E nhiều nhất), **Táo** hoặc **Cà chua**.
- Ba kiểu trả lời, giống `app/inference.py`:
  - năng lượng < τ_E → "Có vẻ không phải lá táo/cà chua";
  - xác suất lớn nhất < τ_c → "Có thể: …, hãy chụp lại" (τ_c của hai mô hình hiện bằng 0, nên kiểu này không xuất hiện);
  - còn lại → tên bệnh.
- Yêu cầu: Android 9 trở lên, CPU ARM (gần như mọi điện thoại). Không cần Internet, không xin quyền nào.
- File cài: `PlantDx-TaoCaChua.apk` (73 MB; phần lớn là thư viện ONNX Runtime cho 2 kiến trúc CPU).
  Mã ứng dụng là `vn.plantdx.species`, cài song song được với ứng dụng 27 lớp cũ.

## Cài lên điện thoại

1. Chép `PlantDx-TaoCaChua.apk` vào điện thoại (cáp USB, Google Drive, Zalo, email…).
2. Mở file trên điện thoại → nếu được hỏi, cho phép ứng dụng đó **"Cài đặt ứng dụng không rõ nguồn gốc"**.
3. Bấm **Cài đặt**. Google Play Protect có thể cảnh báo vì ứng dụng không qua cửa hàng → chọn *Vẫn cài đặt*.

Hoặc qua cáp USB (bật *Gỡ lỗi USB* trong Tuỳ chọn nhà phát triển):

```bash
/opt/homebrew/share/android-commandlinetools/platform-tools/adb install -r PlantDx-TaoCaChua.apk
```

## Tiền xử lý khớp với Python

Mô hình cà chua nhạy với cách thu nhỏ ảnh: đổi bilinear của Pillow sang LANCZOS, BOX hay BICUBIC thì 15–25% ảnh
test đổi kết luận và độ chính xác giảm tới 8 điểm. Vì vậy ứng dụng **không** dùng bộ giải mã Android để resize mà tự
làm đúng các bước của `app/inference.py`:

| Bước | Java | Đã đối chiếu với Python |
|---|---|---|
| Xoay theo EXIF | `ImageDecoder` | — |
| Cạnh ngắn → 255, bilinear có khử răng cưa | `Resample.java` (chép thuật toán libImaging/Resample.c) | 48 ảnh (test PV/PD/PP/TV, ảnh ngẫu nhiên 31×57 đến 3000×4000): trùng từng pixel |
| Cắt giữa 224, chuẩn hoá mean/std | `Preprocess.cropOffset`, `toChw` (làm tròn kiểu Python: `Math.rint`) | 40 ảnh test: tensor trùng tuyệt đối |
| softmax(logits/T*), năng lượng, quyết định 3 mức | `Preprocess` | 40 ảnh test: cùng nhãn, cùng kết luận (36 trả lời, 4 "không phải lá") |

Chỉ có một điểm khác: ảnh lớn hơn 2.048 px (ảnh camera) được bộ giải mã thu về cạnh dài 2.048 trước, cho đỡ tốn bộ nhớ.
Ảnh trong tập dữ liệu không bị ảnh hưởng. Phần chưa kiểm được trên Mac là giải mã JPEG của Android so với Pillow
(có thể lệch ±1 mức xám ở vài pixel).

## Build lại

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
./gradlew assembleRelease
cp app/build/outputs/apk/release/app-release.apk PlantDx-TaoCaChua.apk
```

Cần JDK 21 và Android SDK (platform 35, build-tools); đường dẫn SDK trong `local.properties`.
APK được ký bằng khoá debug để cài trực tiếp; muốn đưa lên Google Play cần tạo khoá ký riêng.
Khi huấn luyện lại, chép `models/<loài>/model.onnx` và `meta.json` vào `app/src/main/assets/<loài>/` rồi build lại.

## Cấu trúc

```
android/
├── PlantDx-TaoCaChua.apk               file cài đặt
├── app/build.gradle.kts                cấu hình (minSdk 28, ONNX Runtime 1.32.0)
└── app/src/main/
    ├── assets/apple/, assets/tomato/   model.onnx (INT8) + meta.json, chép từ models/<loài>/
    └── java/vn/plantdx/
        ├── Resample.java               resize bilinear giống hệt Pillow
        ├── Preprocess.java             cắt giữa, chuẩn hoá, hiệu chỉnh nhiệt độ, năng lượng, quyết định
        ├── SpeciesModel.java           đọc meta.json, chạy ONNX Runtime
        └── MainActivity.java           giao diện: chọn loài, chụp / chọn ảnh, hiển thị kết quả
```
