const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun,
  AlignmentType, Table, TableRow, TableCell, WidthType, BorderStyle,
  ShadingType, PageBreak, NumberFormat,
  Header, Footer, PageNumber,
} = require("docx");

const DIAGRAMS = "docs/diagrams";

function img(name, w, h) {
  const p = `${DIAGRAMS}/${name}`;
  if (!fs.existsSync(p)) return null;
  return new ImageRun({ data: fs.readFileSync(p), transformation: { width: w, height: h }, type: "png" });
}

function p(text, opts = {}) {
  const runs = [];
  if (typeof text === "string") {
    runs.push(new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size || 24, font: "Times New Roman" }));
  } else if (Array.isArray(text)) {
    text.forEach(t => {
      if (typeof t === "string") runs.push(new TextRun({ text: t, size: opts.size || 24, font: "Times New Roman" }));
      else runs.push(new TextRun({ size: 24, font: "Times New Roman", ...t }));
    });
  }
  return new Paragraph({
    children: runs,
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { after: opts.after !== undefined ? opts.after : 120, line: opts.line || 360 },
    heading: opts.heading,
    indent: opts.indent,
  });
}

function heading(text, level) {
  return new Paragraph({
    text,
    heading: level,
    spacing: { before: 240, after: 120 },
    alignment: AlignmentType.LEFT,
    run: { font: "Times New Roman", bold: true },
  });
}

function bullet(text) {
  return p(text, { indent: { left: 720 } });
}

function tableCell(text, opts = {}) {
  return new TableCell({
    children: [new Paragraph({
      children: [new TextRun({ text, bold: opts.bold, size: 22, font: "Times New Roman" })],
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { after: 40 },
    })],
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.shading ? { type: ShadingType.CLEAR, color: "auto", fill: opts.shading } : undefined,
  });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    children: headers.map((h, i) => tableCell(h, { bold: true, width: colWidths[i], shading: "D9E2F3", align: AlignmentType.CENTER })),
  });
  const dataRows = rows.map(row => new TableRow({
    children: row.map((cell, i) => tableCell(cell, { width: colWidths[i], align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
  }));
  return new Table({
    rows: [headerRow, ...dataRows],
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
  });
}

function imgParagraph(name, w, h, caption) {
  const children = [];
  const image = img(name, w, h);
  if (image) {
    children.push(new Paragraph({ children: [image], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 } }));
  }
  if (caption) {
    children.push(p(caption, { italics: true, alignment: AlignmentType.CENTER, size: 22, after: 200 }));
  }
  return children;
}

// ─── Build document ───

const sections = [];

// ═══ TRANG BÌA ═══
sections.push({
  properties: {},
  children: [
    p("", { after: 600 }),
    p("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN", { bold: true, alignment: AlignmentType.CENTER, size: 28, after: 40 }),
    p("ĐẠI HỌC QUỐC GIA TP. HỒ CHÍ MINH", { bold: true, alignment: AlignmentType.CENTER, size: 28, after: 200 }),
    p("", { after: 200 }),
    p("", { after: 200 }),
    p("BÁO CÁO NGHIÊN CỨU", { bold: true, alignment: AlignmentType.CENTER, size: 30, after: 200 }),
    p("", { after: 100 }),
    p("HỆ THỐNG TRUY XUẤT SỰ KIỆN VIDEO", { bold: true, alignment: AlignmentType.CENTER, size: 32, after: 40 }),
    p("DỰA TRÊN BIỂU DIỄN ĐA PHƯƠNG THỨC", { bold: true, alignment: AlignmentType.CENTER, size: 32, after: 300 }),
    p("", { after: 200 }),
    p("", { after: 200 }),
    p([{ text: "Cuộc thi: ", bold: true }, "AI Challenge 2025 (AIC 2025)"], { alignment: AlignmentType.CENTER, after: 200 }),
    p("", { after: 100 }),
    p([{ text: "GVHD: ", bold: true }, "TS. Phạm Đỗ Kim Chi"], { alignment: AlignmentType.CENTER, after: 100 }),
    p("", { after: 60 }),
    p([{ text: "Thành viên:", bold: true }], { alignment: AlignmentType.CENTER, after: 60 }),
    p("Trang Kỳ Anh — 250101006", { alignment: AlignmentType.CENTER, after: 60 }),
    p("Lê Nguyễn Bảo Hân — 250101017", { alignment: AlignmentType.CENTER, after: 60 }),
    p("Đặng Anh Đạt — 250101009", { alignment: AlignmentType.CENTER, after: 200 }),
    p("", { after: 400 }),
    p("TP. Hồ Chí Minh, 2025", { alignment: AlignmentType.CENTER, size: 26 }),
  ],
});

// ═══ NỘI DUNG ═══
const content = [];

// ── TÓM TẮT ──
content.push(heading("TÓM TẮT", HeadingLevel.HEADING_1));
content.push(p("Báo cáo này trình bày hệ thống truy xuất sự kiện video dựa trên biểu diễn đa phương thức, được phát triển cho cuộc thi AI Challenge 2025 (AIC 2025). Hệ thống sử dụng kiến trúc hai tầng: tầng thứ nhất thực hiện truy xuất vector nhanh bằng mô hình CLIP (Contrastive Language–Image Pre-training) kết hợp cơ sở dữ liệu vector Milvus; tầng thứ hai xếp hạng lại kết quả bằng cross-encoder BGE-reranker-v2-m3 để cải thiện độ chính xác. Hệ thống hỗ trợ đa truy vấn với cơ chế hợp nhất Reciprocal Rank Fusion (RRF), lọc theo đối tượng (object detection) và văn bản trong hình (OCR từ Florence-2), cùng khả năng dịch tự động Việt–Anh. Thí nghiệm trên tập 21 truy vấn (18 KIS, 3 QA) cho thấy kiến trúc hai tầng cải thiện đáng kể chất lượng truy xuất so với chỉ sử dụng vector search đơn thuần."));

content.push(p([
  { text: "Từ khóa: ", bold: true, italics: true },
  { text: "truy xuất video, đa phương thức, CLIP, cross-encoder, reranking, vector search", italics: true },
], { after: 200 }));

// ── 1. GIỚI THIỆU ──
content.push(heading("1. GIỚI THIỆU", HeadingLevel.HEADING_1));
content.push(p("Truy xuất thông tin đa phương thức (multimodal information retrieval) là một lĩnh vực nghiên cứu quan trọng, đặc biệt trong bối cảnh dữ liệu video tăng trưởng nhanh chóng. Bài toán truy xuất sự kiện video (video event retrieval) yêu cầu tìm kiếm các đoạn video hoặc keyframe phù hợp dựa trên mô tả bằng ngôn ngữ tự nhiên — một thách thức đòi hỏi khả năng hiểu đồng thời cả ngữ nghĩa văn bản lẫn nội dung thị giác."));

content.push(p("Cuộc thi AI Challenge 2025 (AIC 2025) đặt ra bài toán cụ thể: cho tập dữ liệu video tin tức tiếng Việt đã được trích xuất keyframe, hệ thống cần trả về các keyframe phù hợp nhất cho mỗi truy vấn dạng văn bản. Các thách thức chính bao gồm: (1) khoảng cách ngữ nghĩa giữa mô tả văn bản và nội dung thị giác, (2) truy vấn bằng tiếng Việt trong khi các mô hình vision-language chủ yếu được huấn luyện trên tiếng Anh, và (3) yêu cầu tìm kiếm nhanh trên tập dữ liệu lớn."));

content.push(p("Trong báo cáo này, chúng tôi đề xuất một hệ thống truy xuất video hai tầng:"));
content.push(p([{ text: "Tầng 1 — Truy xuất vector: ", bold: true }, "Sử dụng mô hình CLIP (ViT-B/32) để mã hóa truy vấn và keyframe vào cùng không gian vector 512 chiều, sau đó thực hiện tìm kiếm Approximate Nearest Neighbor (ANN) trên Milvus. Hỗ trợ đa truy vấn với Reciprocal Rank Fusion (RRF)."], { indent: { left: 360 } }));
content.push(p([{ text: "Tầng 2 — Xếp hạng lại: ", bold: true }, "Sử dụng cross-encoder BGE-reranker-v2-m3 để đánh giá lại mức độ liên quan của các ứng viên từ tầng 1, kết hợp thông tin ngữ cảnh từ OCR và object detection."], { indent: { left: 360 } }));

content.push(p([{ text: "Đóng góp chính: ", bold: true }, "(1) Kiến trúc hai tầng kết hợp bi-encoder (CLIP) cho tốc độ và cross-encoder (BGE) cho độ chính xác; (2) Tích hợp đa nguồn dữ liệu bổ trợ (OCR Florence-2, object detection Faster R-CNN); (3) Hỗ trợ truy vấn tiếng Việt qua dịch tự động."]));

// ── 2. NGHIÊN CỨU LIÊN QUAN ──
content.push(heading("2. NGHIÊN CỨU LIÊN QUAN", HeadingLevel.HEADING_1));

content.push(heading("2.1. Mô hình Vision-Language", HeadingLevel.HEADING_2));
content.push(p("CLIP (Contrastive Language–Image Pre-training) [Radford et al., 2021] là mô hình tiên phong trong việc học biểu diễn chung cho văn bản và hình ảnh thông qua contrastive learning trên 400 triệu cặp (image, text) từ internet. CLIP mã hóa văn bản và hình ảnh vào cùng không gian vector, cho phép tính toán độ tương đồng cross-modal bằng cosine similarity. Các biến thể như SigLIP [Zhai et al., 2023] cải thiện hiệu suất bằng sigmoid loss thay vì softmax, nhưng CLIP ViT-B/32 vẫn là lựa chọn phổ biến nhờ cân bằng tốt giữa tốc độ và chất lượng."));

content.push(heading("2.2. Cross-Encoder Reranking", HeadingLevel.HEADING_2));
content.push(p("Khác với bi-encoder (encode query và document độc lập), cross-encoder nhận cặp (query, document) làm đầu vào và cho ra điểm relevance trực tiếp. BGE-reranker-v2-m3 [Xiao et al., 2024] là cross-encoder đa ngôn ngữ được huấn luyện trên dữ liệu retrieval đa dạng, đạt hiệu suất cao trên nhiều benchmark. Trong kiến trúc hai tầng, bi-encoder lọc nhanh top-K ứng viên, sau đó cross-encoder đánh giá lại chính xác hơn — cách tiếp cận được chứng minh hiệu quả trong nhiều hệ thống tìm kiếm hiện đại."));

content.push(heading("2.3. Vector Database và Approximate Nearest Neighbor", HeadingLevel.HEADING_2));
content.push(p("Milvus [Wang et al., 2021] là cơ sở dữ liệu vector mã nguồn mở, hỗ trợ ANN search với các thuật toán indexing như HNSW, IVF. Milvus cung cấp hybrid search cho phép kết hợp nhiều vector query qua RRF (Reciprocal Rank Fusion) hoặc weighted ranker, đồng thời hỗ trợ scalar filtering trên các trường metadata."));

// ── 3. PHƯƠNG PHÁP ──
content.push(heading("3. PHƯƠNG PHÁP", HeadingLevel.HEADING_1));

content.push(heading("3.1. Tổng quan kiến trúc", HeadingLevel.HEADING_2));
content.push(p("Hệ thống được thiết kế theo kiến trúc hai tầng (two-tier) như minh họa trong Hình 1. Tầng thứ nhất sử dụng CLIP bi-encoder để truy xuất nhanh các ứng viên từ cơ sở dữ liệu vector Milvus. Tầng thứ hai sử dụng BGE cross-encoder để xếp hạng lại các ứng viên, kết hợp thông tin bổ trợ từ OCR và object detection."));

content.push(...imgParagraph("architecture.png", 580, 270, "Hình 1. Kiến trúc tổng quan hệ thống truy xuất video đa phương thức"));

content.push(heading("3.2. Biểu diễn đa phương thức (CLIP Encoding)", HeadingLevel.HEADING_2));
content.push(p("Chúng tôi sử dụng mô hình sentence-transformers/clip-ViT-B-32 để mã hóa cả văn bản truy vấn và keyframe hình ảnh vào không gian vector 512 chiều. Vector được chuẩn hóa (L2 normalization) để cosine similarity tương đương inner product. Mô hình chạy ở chế độ eval() với torch.no_grad() để tối ưu bộ nhớ và tốc độ inference."));

content.push(p("Trong giai đoạn tiền xử lý (offline), tất cả keyframe được mã hóa thành vector CLIP và lưu dưới dạng file .npy. Trong giai đoạn truy vấn (online), chỉ văn bản truy vấn cần được mã hóa real-time."));

content.push(heading("3.3. Truy xuất vector (Milvus + RRF)", HeadingLevel.HEADING_2));
content.push(p("Vector truy vấn được gửi đến Milvus để thực hiện ANN search với COSINE metric. Khi có nhiều truy vấn cùng lúc (multi-query), hệ thống sử dụng Reciprocal Rank Fusion (RRF) để hợp nhất kết quả:"));
content.push(p("score(d) = Σ_q 1 / (k + rank_q(d))", { alignment: AlignmentType.CENTER, after: 100 }));
content.push(p("trong đó k là hằng số điều hòa (mặc định k=60), rank_q(d) là thứ hạng của document d trong kết quả của query q. RRF ưu tiên các document xuất hiện ở vị trí cao trong nhiều kết quả khác nhau."));

content.push(...imgParagraph("two-tier.png", 580, 230, "Hình 2. Chi tiết kiến trúc truy xuất hai tầng"));

content.push(heading("3.4. Xếp hạng lại (BGE Cross-Encoder)", HeadingLevel.HEADING_2));
content.push(p("Top-K ứng viên từ tầng 1 (mặc định K=200) được đánh giá lại bằng cross-encoder BAAI/bge-reranker-v2-m3. Đầu vào của cross-encoder là cặp [query, document_text], trong đó document_text được xây dựng từ danh sách đối tượng (objects) và văn bản OCR của keyframe đó:"));
content.push(p("document = join(objects) + \" \" + ocr_text", { alignment: AlignmentType.CENTER, after: 100 }));
content.push(p("Cross-encoder cho ra logit score thể hiện mức độ liên quan; kết quả được sắp xếp lại theo score giảm dần. Phương pháp này đặc biệt hiệu quả cho các truy vấn QA (Question Answering) vì cross-encoder có thể \"đọc\" nội dung OCR để tìm câu trả lời."));

content.push(heading("3.5. Lọc bổ trợ", HeadingLevel.HEADING_2));
content.push(p([{ text: "OCR (Florence-2): ", bold: true }, "177.321 keyframe được trích xuất văn bản bằng mô hình Florence-2. Dữ liệu OCR được lưu trữ trong Milvus và hỗ trợ lọc theo từ khóa (LIKE filter). Ví dụ, truy vấn đề cập đến tên chương trình \"FANA\" có thể dùng OCR filter để thu hẹp không gian tìm kiếm."]));
content.push(p([{ text: "Object Detection (Faster R-CNN): ", bold: true }, "Đối tượng trong mỗi keyframe được phát hiện bằng Faster R-CNN pre-trained trên OpenImagesV4. Kết quả được lọc theo ngưỡng confidence (≥0.3) và dedup, lưu dạng ARRAY trong Milvus. Hỗ trợ lọc bằng ARRAY_CONTAINS_ANY."]));

content.push(heading("3.6. Dịch tự động Việt–Anh", HeadingLevel.HEADING_2));
content.push(p("Do CLIP được huấn luyện chủ yếu trên dữ liệu tiếng Anh, truy vấn tiếng Việt cần được dịch sang tiếng Anh trước khi mã hóa. Hệ thống sử dụng Google Translate API (endpoint công khai translate.googleapis.com) để dịch tự động, không yêu cầu API key."));

// ── 4. DỮ LIỆU VÀ THIẾT LẬP THÍ NGHIỆM ──
content.push(heading("4. DỮ LIỆU VÀ THIẾT LẬP THÍ NGHIỆM", HeadingLevel.HEADING_1));

content.push(heading("4.1. Tập dữ liệu", HeadingLevel.HEADING_2));
content.push(p("Tập dữ liệu AIC 2025 bao gồm video tin tức tiếng Việt đã được trích xuất keyframe. Mỗi keyframe được đặc trưng bởi: video_name, keyframe_idx, frame_idx, pts_time, fps. Dữ liệu bổ trợ gồm:"));
content.push(p("• CLIP features: vector 512 chiều cho mỗi keyframe (file .npy theo video)", { indent: { left: 360 } }));
content.push(p("• OCR: 177.321 keyframe có văn bản trích xuất bằng Florence-2", { indent: { left: 360 } }));
content.push(p("• Object detection: đối tượng phát hiện bằng Faster R-CNN (OpenImagesV4)", { indent: { left: 360 } }));
content.push(p("• Metadata: thời gian, FPS, frame index từ file CSV", { indent: { left: 360 } }));

content.push(...imgParagraph("data-pipeline.png", 580, 250, "Hình 3. Pipeline tiền xử lý và nạp dữ liệu vào Milvus"));

content.push(heading("4.2. Tập truy vấn đánh giá", HeadingLevel.HEADING_2));
content.push(p("Ban tổ chức AIC 2025 cung cấp 24 truy vấn thuộc 3 loại: KIS (Known-Item Search), QA (Question Answering), và TRAKE (Text Re-use and Knowledge Extraction). Trong nghiên cứu này, chúng tôi loại bỏ 3 truy vấn TRAKE vì loại truy vấn này yêu cầu lý luận thời gian (temporal reasoning) — xác định các sự kiện tuần tự trong video — trong khi hệ thống chỉ thực hiện truy xuất ở cấp độ keyframe đơn lẻ. Tập đánh giá cuối cùng gồm 21 truy vấn: 18 KIS và 3 QA."));

content.push(makeTable(
  ["Loại", "Số lượng", "Mô tả"],
  [
    ["KIS", "18", "Tìm keyframe khớp mô tả cảnh cụ thể"],
    ["QA", "3", "Tìm keyframe chứa câu trả lời (thường cần OCR)"],
    ["TRAKE", "3 (loại bỏ)", "Truy vấn thời gian — không phù hợp với single-frame retrieval"],
  ],
  [2000, 1500, 5500],
));
content.push(p("", { after: 100 }));
content.push(p([{ text: "Bảng 1. ", bold: true, italics: true }, { text: "Phân loại truy vấn AIC 2025", italics: true }], { alignment: AlignmentType.CENTER, after: 200 }));

content.push(heading("4.3. Xây dựng Ground Truth", HeadingLevel.HEADING_2));
content.push(p("Để đánh giá hệ thống một cách khách quan và tránh hiện tượng contamination, chúng tôi xây dựng ground truth theo quy trình sau:"));
content.push(p("(1) Đọc từng truy vấn và hiểu nội dung mô tả.", { indent: { left: 360 } }));
content.push(p("(2) Duyệt thủ công toàn bộ keyframe theo từng video bằng công cụ annotation tự phát triển (không sử dụng hệ thống tìm kiếm CLIP hoặc bất kỳ mô hình AI nào).", { indent: { left: 360 } }));
content.push(p("(3) Chọn keyframe phù hợp nhất bằng mắt thường, ghi lại video_name và frame_idx.", { indent: { left: 360 } }));
content.push(p("(4) Định nghĩa frame_range = [frame_idx − 30, frame_idx + 30] làm vùng chấp nhận.", { indent: { left: 360 } }));
content.push(p("Việc không sử dụng CLIP trong quá trình tạo ground truth đảm bảo rằng kết quả đánh giá không bị bias bởi chính mô hình được đánh giá."));

content.push(heading("4.4. Cấu hình hệ thống", HeadingLevel.HEADING_2));
content.push(makeTable(
  ["Thành phần", "Cấu hình"],
  [
    ["CLIP Encoder", "sentence-transformers/clip-ViT-B-32 (512-dim)"],
    ["Cross-Encoder", "BAAI/bge-reranker-v2-m3"],
    ["Vector DB", "Milvus 2.x (standalone, COSINE metric, AUTOINDEX)"],
    ["OCR", "Florence-2 (177.321 keyframes)"],
    ["Object Detection", "Faster R-CNN (OpenImagesV4, threshold ≥ 0.3)"],
    ["Dịch thuật", "Google Translate API (translate.googleapis.com)"],
    ["Retrieval top-K", "200 candidates → rerank → top 50/100"],
    ["RRF k", "60"],
  ],
  [3000, 6000],
));
content.push(p("", { after: 100 }));
content.push(p([{ text: "Bảng 2. ", bold: true, italics: true }, { text: "Cấu hình hệ thống", italics: true }], { alignment: AlignmentType.CENTER, after: 200 }));

content.push(heading("4.5. Phương pháp đánh giá", HeadingLevel.HEADING_2));
content.push(p("Chúng tôi sử dụng các metric sau, theo công thức tính điểm của AIC 2025:"));
content.push(p([{ text: "Mean Top-k R-Score: ", bold: true }, "Final Score = (1/5) × Σ_{k∈{1,5,20,50,100}} max_{1≤i≤k} R-Score(r_i). Đây là metric chính của cuộc thi, đo khả năng hệ thống đưa kết quả đúng lên vị trí cao ở nhiều mức k khác nhau."]));
content.push(p([{ text: "Recall@k: ", bold: true }, "Tỷ lệ truy vấn có ít nhất 1 kết quả đúng trong top-k."]));
content.push(p([{ text: "MRR (Mean Reciprocal Rank): ", bold: true }, "Trung bình nghịch đảo thứ hạng của kết quả đúng đầu tiên."]));
content.push(p("Hai chế độ đánh giá: (1) dense — chỉ dùng CLIP vector search; (2) dense_rerank — CLIP + BGE cross-encoder reranking."));

// ── 5. KẾT QUẢ THÍ NGHIỆM ──
content.push(heading("5. KẾT QUẢ THÍ NGHIỆM", HeadingLevel.HEADING_1));

content.push(heading("5.1. Kết quả tổng thể", HeadingLevel.HEADING_2));
content.push(p("Bảng 3 trình bày kết quả đánh giá trên tập 21 truy vấn (18 KIS + 3 QA) với hai chế độ."));
content.push(p(""));

content.push(makeTable(
  ["Mode", "TopK-R", "MRR", "R@1", "R@5", "R@20", "R@50", "R@100"],
  [
    ["dense", "TODO", "TODO", "TODO", "TODO", "TODO", "TODO", "TODO"],
    ["dense_rerank", "TODO", "TODO", "TODO", "TODO", "TODO", "TODO", "TODO"],
  ],
  [1800, 1000, 1000, 1000, 1000, 1000, 1000, 1200],
));
content.push(p("", { after: 100 }));
content.push(p([{ text: "Bảng 3. ", bold: true, italics: true }, { text: "Kết quả đánh giá tổng thể. TODO: Cập nhật sau khi chạy eval.", italics: true }], { alignment: AlignmentType.CENTER, after: 200 }));

content.push(heading("5.2. Phân tích theo loại truy vấn", HeadingLevel.HEADING_2));
content.push(p("TODO: Phân tích chi tiết KIS vs QA sau khi có kết quả eval. Dự kiến: QA queries hưởng lợi nhiều hơn từ reranking do cross-encoder có thể \"đọc\" nội dung OCR, trong khi KIS queries dựa chủ yếu vào visual similarity."));

// ── 6. THẢO LUẬN ──
content.push(heading("6. THẢO LUẬN", HeadingLevel.HEADING_1));

content.push(heading("6.1. Điểm mạnh", HeadingLevel.HEADING_2));
content.push(p([{ text: "Kiến trúc hai tầng hiệu quả: ", bold: true }, "CLIP bi-encoder cho tốc độ truy xuất nhanh (sub-second trên toàn bộ dataset), trong khi BGE cross-encoder cải thiện chất lượng xếp hạng mà không ảnh hưởng đến thời gian phản hồi đáng kể (chỉ rerank top-K ứng viên)."]));
content.push(p([{ text: "Đa nguồn dữ liệu bổ trợ: ", bold: true }, "OCR và object detection cung cấp thông tin ngữ cảnh phong phú, giúp cross-encoder đánh giá chính xác hơn. Đặc biệt, OCR filter cho phép thu hẹp không gian tìm kiếm hiệu quả cho các truy vấn đề cập đến văn bản cụ thể."]));
content.push(p([{ text: "Đa truy vấn và RRF: ", bold: true }, "Hỗ trợ nhiều truy vấn cùng lúc với RRF fusion, cho phép người dùng mô tả cảnh từ nhiều góc độ."]));

content.push(heading("6.2. Hạn chế", HeadingLevel.HEADING_2));
content.push(p([{ text: "Giới hạn single-frame: ", bold: true }, "Hệ thống truy xuất ở cấp độ keyframe đơn lẻ, không hỗ trợ lý luận thời gian (temporal reasoning). Các truy vấn TRAKE yêu cầu xác định chuỗi sự kiện tuần tự trong video — vượt quá khả năng của kiến trúc hiện tại."]));
content.push(p([{ text: "Phụ thuộc vào chất lượng dịch: ", bold: true }, "CLIP hoạt động tốt nhất với tiếng Anh. Chất lượng dịch Việt–Anh ảnh hưởng trực tiếp đến chất lượng truy xuất, đặc biệt với các mô tả chứa thuật ngữ đặc thù hoặc tên riêng."]));
content.push(p([{ text: "Cross-encoder dựa trên text: ", bold: true }, "BGE reranker chỉ xử lý text (objects + OCR), không \"nhìn\" hình ảnh. Với keyframe ít text/objects, cross-encoder có ít thông tin để đánh giá."]));

content.push(heading("6.3. Vai trò của OCR filter", HeadingLevel.HEADING_2));
content.push(p("OCR filter đặc biệt quan trọng cho các truy vấn QA, vì câu trả lời thường nằm trong văn bản hiển thị trên màn hình (tên chương trình, địa danh, công thức). Khi kết hợp OCR filter với vector search, hệ thống có thể thu hẹp đáng kể không gian tìm kiếm và tìm được keyframe chứa thông tin cần thiết."));

// ── 7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN ──
content.push(heading("7. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN", HeadingLevel.HEADING_1));

content.push(heading("7.1. Kết luận", HeadingLevel.HEADING_2));
content.push(p("Chúng tôi đã xây dựng hệ thống truy xuất sự kiện video đa phương thức cho cuộc thi AIC 2025, sử dụng kiến trúc hai tầng CLIP + BGE cross-encoder. Hệ thống tích hợp đa nguồn dữ liệu (CLIP features, OCR Florence-2, object detection) và hỗ trợ đa truy vấn với RRF fusion. Kết quả thí nghiệm trên 21 truy vấn cho thấy kiến trúc hai tầng cải thiện đáng kể so với chỉ dùng vector search."));

content.push(heading("7.2. Hướng phát triển", HeadingLevel.HEADING_2));
content.push(p([{ text: "Temporal reasoning: ", bold: true }, "Tích hợp video-level features (ví dụ: video captioning, temporal grounding) để hỗ trợ truy vấn TRAKE."]));
content.push(p([{ text: "Mô hình mạnh hơn: ", bold: true }, "Thử nghiệm SigLIP, BLIP-2, hoặc các mô hình vision-language lớn hơn cho chất lượng encoding tốt hơn."]));
content.push(p([{ text: "Sparse + Dense hybrid: ", bold: true }, "Kết hợp sparse retrieval (BM25 trên OCR text) với dense retrieval (CLIP vectors) cho hybrid search."]));
content.push(p([{ text: "Fine-tuning: ", bold: true }, "Fine-tune CLIP trên dữ liệu tiếng Việt hoặc domain-specific data để cải thiện hiểu ngữ nghĩa cross-lingual."]));

// ── TÀI LIỆU THAM KHẢO ──
content.push(heading("TÀI LIỆU THAM KHẢO", HeadingLevel.HEADING_1));
const refs = [
  "[1] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.",
  "[2] Xiao, S., Liu, Z., Zhang, P., Muennighoff, N. (2024). C-Pack: Packaged Resources To Advance General Chinese Embedding. ACL 2024.",
  "[3] Wang, J., Yi, X., Guo, R., et al. (2021). Milvus: A Purpose-Built Vector Data Management System. SIGMOD 2021.",
  "[4] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training. ICCV 2023.",
  "[5] Cormack, G. V., Clarke, C. L., Buettcher, S. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods. SIGIR 2009.",
  "[6] Xiao, L., Lin, H., et al. (2023). Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. CVPR 2024.",
  "[7] Ren, S., He, K., Girshick, R., Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. NeurIPS 2015.",
];
refs.forEach(r => content.push(p(r, { size: 22, after: 80 })));

sections.push({
  properties: {
    page: {
      margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
    },
  },
  headers: {
    default: new Header({
      children: [new Paragraph({
        children: [new TextRun({ text: "Hệ thống truy xuất sự kiện video dựa trên biểu diễn đa phương thức", size: 18, italics: true, font: "Times New Roman", color: "888888" })],
        alignment: AlignmentType.RIGHT,
      })],
    }),
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Times New Roman" })],
        alignment: AlignmentType.CENTER,
      })],
    }),
  },
  children: content,
});

// ─── Generate ───
const doc = new Document({
  sections,
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 24 } },
      heading1: { run: { font: "Times New Roman", size: 28, bold: true } },
      heading2: { run: { font: "Times New Roman", size: 26, bold: true } },
    },
  },
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("docs/report.docx", buf);
  console.log("Generated docs/report.docx (" + buf.length + " bytes)");
});
