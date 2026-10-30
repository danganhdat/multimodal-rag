# Lời thuyết trình — Video RAG đa phương thức

Đọc theo thứ tự 24 slide trong `video_rag_eval_slides.pptx`. Các con số là kết quả của benchmark pilot; hai slide phụ lục dùng để trả lời khi cần xem số liệu đầy đủ.

## Slide 1 — Video RAG đa phương thức

Kính chào thầy cô và các bạn. Nhóm em nghiên cứu cách trả lời câu hỏi trên một kho video lớn. Mục tiêu là tìm được đúng khoảnh khắc, rồi trả lời từ hình ảnh có thể kiểm tra lại. Em sẽ trình bày cách xây hệ thống, cách tạo bộ đánh giá, kết quả so sánh các cấu hình và những lỗi còn gặp.

## Slide 2 — Từ video dài đến truy hồi frame-level

Một câu hỏi thường chỉ liên quan vài giây, trong khi một video có thể dài nhiều phút. Nếu biểu diễn cả video bằng một vector, chi tiết ngắn đó dễ bị lẫn với phần còn lại. Nhóm em chọn cách tách video thành các shot và tìm trên keyframe đại diện. Nhờ vậy, kết quả có ảnh và thời điểm cụ thể để đưa cho mô hình thị giác kiểm tra.

## Slide 3 — Shot boundary → keyframe

Ở bước tiền xử lý, nhóm dùng TransNetV2 để phát hiện ranh giới shot, lấy keyframe đại diện và lọc frame trắng hoặc đen. Kho dữ liệu gồm 873 video L21 đến L30; sau xử lý có 168.749 keyframe. Mỗi frame được gắn ID, vị trí trong video và timestamp. Đây là đơn vị mà các bước truy hồi phía sau sẽ tìm kiếm.

## Slide 4 — 5 tầng: offline nặng, online nhẹ

Sơ đồ này chia hệ thống thành hai phần. Offline, nhóm trích keyframe, mã hóa ảnh bằng SigLIP2, lấy chữ bằng OCR và nhãn đối tượng, rồi lập chỉ mục trên Milvus và BM25. Online, hệ thống phân tích câu hỏi, truy hồi ứng viên, xử lý thứ tự thời gian nếu có, xếp hạng và đưa bằng chứng cho VQA hoặc agent. Các phép tính nặng được làm trước khi người dùng đặt câu hỏi.

## Slide 5 — Hiểu truy vấn tiếng Việt

Câu hỏi tiếng Việt có thể chứa nhiều hành động, chữ trên ảnh hoặc quan hệ “trước–sau”. Nhóm dùng LLM để tách các sự kiện và quan hệ thời gian. Song song, quy tắc trích các cụm chữ OCR và tên đối tượng cần tìm chính xác. Nếu LLM phân tích lỗi, hệ thống dịch câu hỏi và dùng một sự kiện cơ bản để vẫn thực hiện truy hồi.

## Slide 6 — Truy hồi lai và RRF

Với mỗi truy vấn, SigLIP2 tạo vector văn bản để tìm frame gần nghĩa trong Milvus; BM25 tìm frame có chữ OCR khớp. Hai danh sách được gộp bằng Reciprocal Rank Fusion, viết tắt là RRF, rồi có thể cộng thêm tín hiệu metadata. Lý do dùng hai nhánh là câu hỏi đôi khi mô tả cảnh bằng nghĩa, đôi khi nhắc đúng một dòng chữ xuất hiện trong video.

## Slide 7 — Temporal chain matching

Temporal search dùng cho câu hỏi có nhiều sự kiện theo thứ tự, chẳng hạn “sau cảnh biển, người ta làm gì ở chợ?”. Nếu chỉ lấy frame biển hoặc frame chợ đứng đầu, ta chưa biết chúng có thuộc cùng video và xuất hiện đúng thứ tự hay không.

Hệ thống tách câu hỏi thành từng sự kiện, truy hồi một danh sách frame cho mỗi sự kiện, rồi nhóm các ứng viên theo video. Với một frame của sự kiện đầu làm điểm bắt đầu, thuật toán chọn frame liên quan nhất cho sự kiện kế tiếp, nhưng chỉ nhận frame có `pts_time` lớn hơn và khoảng cách thời gian không quá ngưỡng. Nó lặp lại bước đó để tạo một chuỗi. Điểm chuỗi bằng độ liên quan trung bình nhân với tỷ lệ sự kiện được tìm thấy; chuỗi thiếu sự kiện sẽ bị giảm điểm. Sau khi bỏ chuỗi trùng, hệ thống giữ các chuỗi tốt nhất làm bằng chứng. Đây là cách ghép greedy: nhanh, nhưng có thể sai khi một cảnh giống nhau lặp lại nhiều lần.

## Slide 8 — Manual và Agentic: chung core, khác luồng

Hai mode dùng chung SigLIP2Encoder, Milvus và hàm truy hồi lõi; feedback cùng trace cũng được ghi trên hạ tầng này. Nhánh one-shot Manual qua `/api/v1/vqa` chạy pipeline cố định: QE v2 phân tích câu hỏi thành QueryPlan, tìm nhiều biến thể rồi gộp RRF, cộng tín hiệu metadata, ghép sự kiện theo thời gian khi cần, dùng MLLM rerank và cuối cùng VQA trả lời một lần. Nhánh Agentic qua `/api/v1/chat` là vòng ReAct: mô hình quyết định gọi `search` hoặc `search_events`, rồi dùng `look`, `crop` hay `neighbors` để kiểm tra frame. Tool `final_answer` đi qua guard, sau đó kết quả được stream bằng SSE. Điểm khác biệt là agent tự quyết định lượt tìm và xem tiếp theo. Nếu người dùng đã chọn sẵn frame, hệ thống còn có `/api/v1/vqa/answer` để trả lời trực tiếp trên các frame đó.

## Slide 9 — One-shot VQA: ảnh + OCR

Ở chế độ one-shot, hệ thống chọn các keyframe đầu bảng và gửi ảnh cùng chữ OCR trong một prompt cho mô hình thị giác. Mô hình phải trả lời dựa trên những gì ảnh thực sự cho thấy. Nếu thiếu bằng chứng, guard yêu cầu nói không xác định được. Cấu hình này giúp đo chất lượng khi mô hình chỉ có một lượt xem context.

## Slide 10 — Agent ReAct: search → look → guard

Agent dùng vòng lặp ReAct: tìm, xem, rồi quyết định có cần tìm tiếp không. Nó có thể mở frame, crop vùng quan trọng hoặc xem frame lân cận trước khi trả lời. Số lượt công cụ được giới hạn để kiểm soát chi phí. Guard cuối cùng chỉ cho phép viện dẫn những frame agent đã thực sự xem. Cách này đặc biệt có ích khi vài kết quả đầu rất giống nhau.

## Slide 11 — Kết quả chính của hệ thống

Trên benchmark pilot gồm 20 câu hỏi và 416 nhãn frame, cấu hình agent E8 đạt Visual Context Recall 0,694, so với 0,528 ở baseline E1. Faithfulness nhị phân của E8 là 0,929. Đổi lại, độ trễ trung vị là 54,5 giây. Các số này được đo trên 14 câu test, nên em xem đây là tín hiệu ban đầu về chất lượng evidence, chưa phải kết luận thống kê cho cả kho video.

## Slide 12 — Tạo benchmark silver

Để đánh giá, nhóm em cần câu hỏi, đáp án và frame làm bằng chứng tham chiếu. Trước hết, pipeline kiểm tra ID và timestamp của 128 video pilot. Nó gom frame trùng hoặc gần trùng bằng hash, pHash, SSIM và độ giống vector SigLIP2, đồng thời so OCR để tránh gộp hai ảnh có chữ khác nhau. Bước này chỉ chọn frame đại diện cho việc lấy mẫu, **không xóa frame khỏi kho**. Từ 120 frame mẫu, mô hình đa phương thức trích các fact nhìn thấy được, rồi sinh mô tả, câu hỏi và đáp án.

Các cặp hỏi đáp chưa được dùng ngay. Hai lượt judge được gọi độc lập trên cùng checkpoint: mỗi lượt xem ảnh nguồn và câu hỏi, tự đưa ra đáp án **trước khi** pipeline so với đáp án tham chiếu. Judge còn kiểm tra câu hỏi có rõ, bằng chứng có đủ và mô tả hoặc câu hỏi có vô tình lộ đáp án không. Cờ `leaks_answer` đã gắn cho 2 đề xuất; proposal không đạt hoặc bất đồng bị loại. Trong 43 đề xuất được judge, 25 được chấp nhận; nhóm chọn 20 để đóng băng benchmark. Cần phân biệt: bước này kiểm tra **lộ đáp án trong văn bản**, chưa phải phép thử hỏi model không kèm ảnh để xem nó có tự đoán đúng hay không.

Với mỗi cặp đã chọn, pipeline dùng **truy vấn văn bản tiếng Anh sinh cùng câu hỏi** để tìm ảnh bằng SigLIP2. Nó bổ sung frame có cùng nhãn đối tượng với ảnh nguồn và một số frame chọn ngẫu nhiên; ảnh nguồn cũng luôn có trong pool. Sau đó, judge xem **câu hỏi và từng ảnh ứng viên** để chấm mức liên quan từ 0 đến 3, nhưng không thấy điểm hay thứ hạng truy hồi: 3 là bằng chứng trực tiếp, 2 là hỗ trợ đáng kể, 1 chỉ cùng bối cảnh, 0 là đã xem và không liên quan. Kết quả có 416 qrels, trong đó 116 nhãn dương, tức grade từ 2 trở lên.

Cuối cùng, nhóm chia câu hỏi theo video thành 6 dev và 14 test, lưu câu hỏi, đáp án, evidence và qrels dưới cùng `query_id`. Bộ này gọi là *silver* vì mô hình tạo và kiểm tra nhãn, chưa có chuyên gia xác nhận. Một frame chưa được judge là chưa biết, không được tự coi là frame sai.

## Slide 13 — Từ nhãn frame đến metric

Bộ eval không có một file ground truth duy nhất. `queries.jsonl` chứa câu hỏi; `answers.jsonl` chứa đáp án tham chiếu; `evidence.jsonl` giữ các frame hỗ trợ; `qrels.tsv` chấm độ liên quan của từng frame. Từ qrels và ánh xạ keyframe, nhóm suy ra `accepted_ranges.tsv` để chấm cả những frame lân cận hợp lệ. Retrieval được so với qrels và khoảng chấp nhận; VQA được chấm trên ảnh evidence mà mô hình thực sự nhận.

## Slide 14 — Bốn metric cốt lõi

Nhóm dùng bốn metric bổ trợ nhau. Visual Context Recall hỏi các claim trong câu trả lời có đủ ảnh hỗ trợ không. Context Precision hỏi frame hữu ích có nằm sớm trong danh sách context không. RAGAS MultiModalFaithfulness đưa ra verdict nhị phân về việc câu trả lời có bám evidence. RAGAS Answer Relevancy đo câu trả lời có đúng trọng tâm câu hỏi; nó không phải thước đo độ đúng của từng fact.

## Slide 15 — Giao thức và giới hạn diễn giải

Hai mươi câu được chia theo video và nhóm gần trùng: 6 dev, 14 test. Pool 416 qrels chỉ là một phần của kho, nên frame chưa chấm không đồng nghĩa với không liên quan. Generator và judge dùng cùng checkpoint với prompt khác nhau, vẫn có nguy cơ cùng thiên lệch. Ngoài ra, E1 đến E8 đánh giá nhánh dense SigLIP2; các số ablation chưa đo đầy đủ BM25 hybrid và MLLM rerank của pipeline production.

## Slide 16 — E1–E8: mỗi bước kiểm tra một thành phần

E1 là mốc dense SigLIP2. E2 thêm query variants và RRF; E3 thêm object boost. E4 thử SuperGlobal để xếp hạng lại 50 frame đầu: với mỗi frame, nó lấy trung bình embedding của frame đó và ba frame gần nó nhất trong nhóm, rồi tính lại độ giống với truy vấn. Ý tưởng là giảm nhiễu ở một frame riêng lẻ; bước này chỉ tính trên vector, không gọi thêm mô hình thị giác. E5 thêm crop ảnh, E6 thêm evidence theo thời gian, E7 dùng tóm tắt shot, và E8 dùng agent ReAct. Các cấu hình giúp kiểm tra từng hướng cải thiện, dù không phải mọi cặp đều chỉ thay đúng một biến.

## Slide 17 — Retrieval: RRF giúp, SuperGlobal hại

Từ E1 sang E2, Recall@10 tăng từ 0,392 lên 0,458; MRR@10 tăng từ 0,417 lên 0,536. Nghĩa là query variants cộng RRF giúp đưa frame đúng lên gần đầu danh sách hơn. E3 gần như không đổi ở hai metric này. E4 giảm MRR@10 xuống 0,326, nên SuperGlobal chưa phù hợp với tập pilot và cách xếp hạng hiện tại.

## Slide 18 — VQA: agent cải thiện chất lượng context

E8 có Visual Context Recall 0,694, Context Precision 0,603 và Faithfulness 0,929 — cao nhất trong bảng này. Điều đó cho thấy việc agent tìm rồi xem lại frame giúp context sát câu hỏi hơn và câu trả lời bám context hơn. Tuy nhiên, Answer Correctness vẫn là 0,250. Nói cách khác, evidence tốt hơn chưa tự động giải quyết mọi lỗi về đáp án.

## Slide 19 — Đánh đổi chất lượng ↔ độ trễ

Chất lượng evidence cao hơn đi kèm thời gian xử lý. E2 có độ trễ trung vị 9,8 giây; E5 dùng crop là 42,7 giây; E7 là 52,9 giây; E8 agent là 54,5 giây. E5 đạt Answer Relevance 0,796, còn E8 dẫn đầu các metric về context. Với Answer Correctness chỉ từ 0,167 đến 0,250 trên 14 câu test, nhóm chưa thể chọn một cấu hình chỉ theo một con số.

## Slide 20 — Lỗi xảy ra ở nhiều tầng

Lỗi không chỉ đến từ mô hình trả lời. Retrieval có thể bỏ sót cảnh do cách diễn đạt khác nhau; OCR có thể đọc thiếu chữ; bước chọn top-3 có thể lấy nhầm frame. Dịch Việt–Anh hoặc ghép chuỗi thời gian cũng có thể làm lệch ý. Cuối cùng, ngay cả khi tìm đúng ảnh, VLM vẫn có thể suy đoán hoặc không đọc được chữ Hán-Nôm và chữ viết tay. Vì vậy, cần xem từng tầng để biết sửa ở đâu.

## Slide 21 — Hai ca lỗi cho thấy giới hạn thực tế

Ca đầu là bài thơ ở đình Nguyễn Trung Trực. Hệ thống tìm đúng ảnh, nhưng Florence-2 và VLM không đọc được chữ thư pháp Hán-Nôm; agent crop thêm vẫn không giải quyết được. Hướng xử lý là OCR chuyên biệt hoặc từ chối trả lời khi chưa đủ căn cứ. Ca thứ hai hỏi món có 200 gram thịt xay. OCR “200g” xuất hiện ở nhiều công thức nên top-3 ban đầu chọn nhầm. Agent mở frame `L30_V078/031`, đọc được “Nhân bánh cuốn” và sửa được lựa chọn evidence.

## Slide 22 — Retrieval ablation · đầy đủ E1–E8

Bảng này cho số retrieval đầy đủ ở các ngưỡng 10, 50 và 300. Điểm đáng chú ý vẫn là E2 cải thiện vị trí đầu danh sách so với E1, còn E4 làm MRR và nDCG giảm. E5 đến E8 dùng cùng kết quả retrieval đầu vào với E2; khác biệt của chúng nằm ở bước xử lý evidence và trả lời sau đó. Các số được tính trên 14 câu test.

## Slide 23 — VQA ablation · đầy đủ E1–E8

Bảng cuối cho thấy vì sao phải đọc nhiều metric cùng lúc. E8 tốt nhất về chất lượng context, nhưng Answer Correctness vẫn chỉ 0,250. E4 có Faithfulness 1,0, trong khi Visual Context Recall là 0,333 và Context Precision là 0,108: câu trả lời có thể bám vào context đang có, dù context đó chưa đủ tốt. Tập test nhỏ nên các chênh lệch cần kiểm chứng thêm.

## Slide 24 — Điều cần giữ và bước tiếp theo

Kết quả pilot ủng hộ cách lập chỉ mục ở mức frame và việc kiểm tra ảnh trước khi trả lời. Nhóm sẽ giữ kiến trúc kết hợp dense, BM25, RRF và evidence guard; agent phù hợp khi cần xem lại bằng chứng. Bước tiếp theo là mở rộng lên hơn 300 câu hỏi, kiểm tra nhãn bằng con người, cải thiện OCR chuyên biệt và cách chọn chuỗi theo thời gian. Em xin cảm ơn thầy cô và các bạn.
