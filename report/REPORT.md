# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/09/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  class_id: 468
  class_name: "cab"
  rank: 1
  score: 0.510915
  taxonomy_name: "ImageNet-1K"
- Record này mô tả toàn ảnh như thế nào?
  Record trên cho biết YOLO11 Classification dự đoán toàn bộ ảnh thuộc lớp cab (taxi). Model đánh giá đây là lớp có xác suất cao nhất với score = 0.510915. Điều này không có nghĩa ảnh chỉ có cab(taxi), mà cab(taxi) là đối tượng nổi bật nhất theo mô hình.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Class list do bộ dữ liệu dùng để huấn luyện checkpoint định nghĩa và được lưu trong checkpoint. Danh sách đó là các lớp thuộc taxonomy ImageNet-1K; mô hình chỉ có thể dự đoán những lớp đã học trong danh sách này.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  + ID: vì nó là mã định danh dùng để phân biệt các lớp khác nhau trong cùng một hệ phân loại.
  + tên lớp: để giúp con người đọc và dự đoán đó là lớp gì.
  + taxonomy: cho biết 1 lớp thuộc hệ phân loại nào. Ví dụ có 2 lớp phân loại ImageNet-1k và ImageNet-2k, nếu có 2 lớp trùng tên hoặc ID và không để tên taxonomy có thể gây nhầm lẫn.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Nếu ảnh có nhiều chủ thể, guideline cần quy định rõ:
   + Nhãn cấp ảnh phải đại diện cho chủ thể/cảnh nào: chủ thể chính, lớn nhất, nổi bật nhất hay toàn cảnh.
   + Khi nào cần gán nhiều nhãn hoặc chuyển sang detection/segmentation thay vì chỉ một nhãn.
   + Cách xử lý chủ thể nhỏ, bị che khuất, sát mép ảnh và các đối tượng phụ.
   + Yêu cầu nhận diện riêng cho từng chủ thể (trang phục, đặc điểm, màu sắc).
- Vì sao model score không phải ground truth?
  Model score là mức độ tin cậy do model tự ước lượng cho prediction, không phải nhãn đúng đã được con người xác nhận. 

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  class_name: person
  score: 0.912625
  bbox_xyxy: [385.33, 69.24, 498.92, 348.92]
  bbox_width: 113.58 pixels
  bbox_height: 279.68 pixels
  Record này cho biết model phát hiện một người với độ tin cậy khoảng 91,26%; box trải từ điểm (385.33, 69.24) ở góc trên-trái đến (498.92, 348.92) ở góc dưới-phải.
- Diễn giải vị trí box bằng lời: Box bao quanh một người ở phía bên phải ảnh kitchen, kéo dài từ gần phía trên xuống gần phía dưới ảnh. Cụ thể, box bắt đầu tại (385.33, 69.24) và kết thúc tại (498.92, 348.92), nên đối tượng có chiều cao nổi bật hơn chiều rộng.
- So sánh số prediction ở hai threshold: Với sample kitchen, ở threshold 0.35 có 11 prediction. Nếu nâng threshold lên 0.60, chỉ còn 6 prediction có score từ 0.60 trở lên.
Như vậy, tăng threshold từ 0.35 lên 0.60 đã loại 5 prediction có độ tin cậy thấp hơn; số box cần xem giảm, nhưng cũng có nguy cơ bỏ sót các vật thể thật.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? 
  + Khi hạ threshold, model giữ nhiều prediction hơn nên độ bao phủ đối tượng có thể tăng, giảm nguy cơ bỏ sót vật thể. Đổi lại reviewer phải xem nhiều box hơn, gồm cả các prediction kém tin cậy hoặc sai.
  + Khi nâng threshold, số box giảm nên reviewer đỡ việc, nhưng độ bao phủ giảm và có thể bỏ sót các đối tượng thật có score thấp. Với ảnh kitchen, tăng ngưỡng từ 0.35 lên 0.60 giảm từ 11 còn 6 prediction: reviewer xem ít hơn 5 box, nhưng cần chấp nhận rủi ro mất các object đó.
- Đề xuất một quy tắc box chặt: Mỗi object nhìn thấy được phải có đúng một box theo lớp đã quy định; box phải ôm sát phần nhìn thấy của object, gồm toàn bộ phần đó nhưng không lấy thêm nền hoặc object khác. Box được ghi theo định dạng xyxy với (x1, y1) là góc trên-trái và (x2, y2) là góc dưới-phải; không được có chiều rộng hoặc chiều cao bằng/nhỏ hơn 0. Nếu hai object cùng lớp tách biệt, gán hai box riêng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
 Guideline cần quy định box chỉ bao phần object còn nhìn thấy hay được phép ước lượng phần bị che khuất/nằm ngoài ảnh; đồng thời quy định ngưỡng phần nhìn thấy tối thiểu để gán nhãn và cách gắn cờ occluded/truncated.
 Nếu không xác định được lớp, ranh giới phần nhìn thấy hoặc object có đủ điều kiện gán nhãn, annotator phải đánh dấu trường hợp mơ hồ và escalation cho reviewer quyết định thống nhất.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  instance_id: kitchen-001
  class_name: person
  score: 0.899318
  polygon_point_count: 348
  Một phần polygon_xy: [[446, 70], [445, 71], [444, 71], [443, 72], [442, 72], [441, 73], ...]
  Record này biểu diễn một instance người; 348 điểm tọa độ nối tiếp nhau tạo thành đường biên mask của người đó.
- Polygon bổ sung chi tiết gì so với box? Polygon mô tả chính xác đường biên và hình dạng phần pixel thuộc từng instance, ví dụ theo cơ thể người hoặc viền vật dụng. Box chỉ là hình chữ nhật bao ngoài nên thường chứa cả nền và các vùng không thuộc object. Vì vậy polygon phù hợp khi cần biết vùng object cụ thể, xử lý object chồng lấp hoặc tính diện tích chính xác.
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id dùng để định danh duy nhất từng object cụ thể trong một ảnh/kết quả segmentation, ví dụ kitchen-001 là một người riêng biệt. Nó giúp liên kết polygon, box, lớp và score của đúng instance đó.
- Đề xuất một quy tắc biên mask: Mask phải bám theo ranh giới phần object nhìn thấy được ở mức pixel: bao gồm toàn bộ pixel thuộc object, không lấn sang nền hay object khác. Các lỗ/rỗng thật bên trong object phải được giữ lại; các chi tiết quá nhỏ, mờ hoặc không xác định được ranh giới thì áp dụng quy tắc làm mượt đã định nghĩa và gắn cờ để reviewer kiểm tra.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Guideline cần quyết định:
  + Mask chỉ bao phần nhìn thấy hay được ước lượng phần bị che khuất.
  + Cách vẽ biên tại vùng mờ/khó phân biệt.
  + Cách tách thành các instance riêng khi hai object tiếp xúc hoặc chồng lên nhau.
  + Ngưỡng diện tích/phần nhìn thấy tối thiểu để giữ hoặc bỏ một object.
Nếu các quy tắc này vẫn không cho kết quả rõ ràng, annotator phải gắn cờ và escalation cho reviewer để đưa ra quyết định thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

- Phân loại ảnh:
  + Đơn vị/định dạng ground truth: Một hoặc nhiều nhãn lớp cho toàn ảnh, theo taxonomy đã chọn
  + Lỗi hoặc điểm mơ hồ quan sát được: nhãn lớp cho toàn ảnh, theo taxonomy đã chọnẢnh nhiều chủ thể; prediction cab hoặc gong có thể không phản ánh đầy đủ nội dung ảnh
  + Annotator làm gì?: Chọn nhãn theo guideline, xác định chủ thể/cảnh chính và gắn cờ trường hợp mơ hồ
  + Reviewer xem gì? Nhãn đúng taxonomy, có nhất quán với nội dung ảnh và quy tắc nhiều chủ thể không
- Phát hiện vật thể:
  + Đơn vị/định dạng ground truth: Mỗi object là một class_id/class_name và box xyxy
  + Lỗi hoặc điểm mơ hồ quan sát được: Box quá rộng/hẹp, trùng object, bỏ sót object nhỏ, che khuất hoặc cắt mép
  + Annotator làm gì?: Gán một box sát phần nhìn thấy cho từng object đủ điều kiện
  + Reviewer xem gì? Lớp, số lượng object, tọa độ/độ khít box, xử lý object che khuất/cắt mép.
- Instance segmentation:
  + Đơn vị/định dạng ground truth: Mỗi instance có lớp, instance_id và polygon/mask
  + Lỗi hoặc điểm mơ hồ quan sát được: Biên mask lấn nền, thiếu phần object, khó tách object tiếp xúc/che khuất hoặc vùng mờ
  + Annotator làm gì?: Vẽ mask theo ranh giới pixel nhìn thấy, tách từng instance và gắn cờ vùng mơ hồ
  + Reviewer xem gì? Độ chính xác biên mask, không chồng/lẫn instance, tính nhất quán khi che khuất hoặc tiếp xúc

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Chỉ sử dụng ảnh và annotation trong phạm vi được cấp quyền; không tải lên, chia sẻ hoặc đưa vào báo cáo bất kỳ dữ liệu cá nhân/nhạy cảm nào, và phải xóa hoặc báo cáo ngay dữ liệu ngoài phạm vi theo quy trình dự án.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho giảng viên/người phụ trách dự án hoặc quản trị dữ liệu theo quy trình được quy định.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
