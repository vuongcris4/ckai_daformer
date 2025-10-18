# deploy.py (phiên bản hoàn chỉnh)
import numpy as np, cv2, os, io, tempfile, re
from PIL import Image
import gradio as gr
import mmcv

from tools.test import update_legacy_cfg
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.core.evaluation import get_classes, get_palette

# ====== THIẾT LẬP CỦA BẠN ======
DEVICE = "cuda:0"  # hoặc "cpu"

CONFIG = "work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/211108_1622_gta2cs_daformer_s0_7f24c.json"
CKPT   = "work_dirs/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb3_s0/latest.pth"

PALETTE_NAME = 'cityscapes'
CLASSES = get_classes(PALETTE_NAME)
PALETTE = get_palette(PALETTE_NAME)

# ====== MODEL ======
def build_model():
    """Khởi tạo model segmentation."""
    cfg = mmcv.Config.fromfile(CONFIG)
    cfg = update_legacy_cfg(cfg)
    model = init_segmentor(
        cfg,
        CKPT,
        device=DEVICE,
        classes=CLASSES,
        palette=PALETTE,
        revise_checkpoint=[(r'^module\.', ''), (r'^model\.', '')]
    )
    model.CLASSES = tuple(CLASSES)
    model.PALETTE = PALETTE
    return model

MODEL = build_model()

# Tạo bảnh màu gôm n màu ngẫu nhiên (để hiển thị mỗi class một màu khác nhau)
def random_color_map(n=256, seed=0):
    rs = np.random.RandomState(seed)
    return (rs.rand(n,3)*255).astype(np.uint8)

COLOR_MAP = random_color_map(256, seed=0)  # (256, 3, mỗi hàng là một màu RGB 8 bit)

def to_pil(arr):    # để có thể dùng các hàm của Object PIL Image
    return Image.fromarray(arr)

def save_temp_png(arr):
    """Lưu array (H,W[,C]) thành PNG tạm, trả path để DownloadButton dùng."""
    img = Image.fromarray(arr)
    fd, path = tempfile.mkstemp(suffix=".png")  # (file descriptor, path)
    os.close(fd)
    img.save(path)
    return path

# ====== 1) CHẠY SEGMENTATION ======
"""
Nhận ảnh đầu vào (PIL Image), Chạy model segmentation
"""
def run_segmentation(image_pil):
    img = np.array(image_pil.convert("RGB")) # PIL Image -> (R, G, B) -> Chuyển qua NumPy (HxW×3)
    result = inference_segmentor(MODEL, img)   # API MMSeg reisze ảnh đúng kích thước model, đưa qua model, nhận đầu ra là mask phân lớp cho từng pixel
    seg = result[0] if isinstance(result, (list,tuple)) else result
    overlay = (0.5*img + 0.5*COLOR_MAP[seg]).astype(np.uint8)    # HxW seg là mảng index, thay các color (3 channels) vào mảng seg
    return to_pil(overlay), seg, img

# ====== 2) RENDER LẠI OVERLAY THEO CLASS/OPACITY ======

# Overy theo class được chọn/ Độ đậm tuỳ chỉnh, chỉ hiển thị các class được chọn
"""
img_np → ảnh RGB gốc, shape (H, W, 3)
seg → ảnh 1 kênh chứa ID class (0,1,2,...)
COLOR_MAP → mảng (num_classes, 3) chứa màu RGB cho từng class
vis_mask → mặt nạ boolean (H, W) chỉ định pixel nào được phủ màu
opacity → độ trong suốt (giá trị 0-1)
"""
def render_overlay(img_np, seg, visible_classes, opacity):
    if img_np is None or seg is None:    # ảnh gốc (H, W, 3)
        return None
    vis_mask = np.zeros(seg.shape, dtype=bool)  # mặc định false
    # Nếu không chọn class nào, hiển thị tất cả
    if not visible_classes:
        visible_classes = CLASSES

    for cname in visible_classes:
        if cname in CLASSES:
            cid = CLASSES.index(cname) # tìm id của visible_classes
            vis_mask |= (seg==cid)  # Cộng dồn phép or, trả về mặt nạ

    color = COLOR_MAP[seg]
    out = img_np.copy() # Tạo bản sao của ảnh gốc, tránh thay đổi img_np.
    # Dòng này chỉ cập nhật những pixel có vis_mask == True.
    # Trộn alpha giữa ảnh gốc và màu class với độ mờ opacity
    out[vis_mask] = ((1-opacity)*img_np[vis_mask] + opacity*color[vis_mask]).astype(np.uint8) # opacity=1 -> chỉ màu class, opacity=0 -> chỉ ảnh gốc
    return to_pil(out)

# ====== 3) CLICK → TÁCH COMPONENT THEO ĐIỂM ======
def click_extract(evt: gr.SelectData, image_pil, seg, min_area, feather):
    if image_pil is None or seg is None or evt is None:
        return None, None, "Hãy chạy segmentation rồi click lên ảnh.", None, None, None
    x, y = evt.index
    img = np.array(image_pil.convert("RGB"))
    h, w = seg.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None, None, "Điểm click ngoài ảnh.", None, None, None

    cid = int(seg[y, x])
    cname = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"class_{cid}"
    mask = (seg == cid).astype(np.uint8)
    
    nlabel, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)    
    lab_id = labels[y, x]
    area = int(stats[lab_id, cv2.CC_STAT_AREA])
    if area < min_area:
        return None, None, f"Vùng quá nhỏ (<{min_area}). Class: {cname}", None, None, None

    comp = (labels == lab_id).astype(np.uint8)
    if feather > 0:
        comp = cv2.GaussianBlur(comp.astype(np.float32), (0,0), feather)    
        comp = np.clip(comp, 0, 1)

    alpha = (comp*255).astype(np.uint8)
    rgba = np.dstack([img, alpha])
    x0,y0,wc,hc = stats[lab_id,0], stats[lab_id,1], stats[lab_id,2], stats[lab_id,3]
    rgba_crop = rgba[y0:y0+hc, x0:x0+wc]
    output_pil = to_pil(rgba_crop)
    info = f"Class: {cname} (id={cid}) • Area: {area} px • BBox: {wc}×{hc}"

    # --- LOGIC MỚI: TẠO ẢNH CANVAS ĐỂ FIT VỚI SKETCHPAD ---
    canvas_w, canvas_h = 540, 360
    original_w, original_h = output_pil.size
    ratio = min(canvas_w / original_w, canvas_h / original_h)
    new_w, new_h = int(original_w * ratio), int(original_h * ratio)
    resized_img = output_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    output_pil_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    paste_x = (canvas_w - new_w) // 2
    paste_y = (canvas_h - new_h) // 2
    output_pil_canvas.paste(resized_img, (paste_x, paste_y), resized_img)

    # Đóng gói thông tin cần thiết để truyền đi
    crop_info = {
        "paste_x": paste_x, "paste_y": paste_y,
        "new_w": new_w, "new_h": new_h,
        "original_w": original_w, "original_h": original_h
    }

    # CẬP NHẬT: Trả về thêm output_pil_canvas để lưu vào state
    return output_pil, output_pil_canvas, info, rgba_crop, crop_info, output_pil_canvas


# ====== 4) ÁP DỤNG SỬA MASK (SKETCH) LÊN OBJECT ======
# CẬP NHẬT: Toàn bộ hàm này đã được thay thế bằng thuật toán mới
def apply_sketch(stashed_rgba_obj, sketch_data, crop_info, canvas_bg_pil):
    """
    Áp dụng nét vẽ từ Sketchpad (Trắng=Thêm, Đen=Xóa) vào alpha channel của object.
    """
    if stashed_rgba_obj is None or crop_info is None or canvas_bg_pil is None:
        return None, None

    rgba = stashed_rgba_obj.copy()
    alpha = rgba[..., 3]

    if sketch_data is not None:
        try:
            composite_np = sketch_data['composite']
            # Chuyển canvas background từ PIL sang NumPy array, đảm bảo có RGBA
            canvas_bg_np = np.array(canvas_bg_pil.convert("RGBA"))

            # --- THUẬT TOÁN MỚI: SO SÁNH TRỰC TIẾP TRẠNG THÁI TRƯỚC VÀ SAU KHI VẼ ---
            comp_rgb = composite_np[..., :3]
            bg_rgb = canvas_bg_np[..., :3]

            # Vùng "thêm" là những pixel TRỞ THÀNH màu trắng (>250)
            is_white = np.all(comp_rgb > 250, axis=-1)
            was_white = np.all(bg_rgb > 250, axis=-1)
            add_mask_canvas = is_white & ~was_white

            # Vùng "xóa" là những pixel TRỞ THÀNH màu đen (<5)
            is_black = np.all(comp_rgb < 5, axis=-1)
            was_black = np.all(bg_rgb < 5, axis=-1)
            erase_mask_canvas = is_black & ~was_black
            # --- KẾT THÚC THUẬT TOÁN MỚI ---
            
            # Lấy thông tin đã lưu
            px, py = crop_info["paste_x"], crop_info["paste_y"]
            nw, nh = crop_info["new_w"], crop_info["new_h"]
            ow, oh = crop_info["original_w"], crop_info["original_h"]

            # 1. Cắt vùng mask từ canvas lớn
            add_mask_cropped = add_mask_canvas[py:py+nh, px:px+nw]
            erase_mask_cropped = erase_mask_canvas[py:py+nh, px:px+nw]

            # 2. & 3. Resize các mask này về kích thước gốc của vật thể
            add_pil = Image.fromarray(add_mask_cropped.astype(np.uint8) * 255)
            erase_pil = Image.fromarray(erase_mask_cropped.astype(np.uint8) * 255)

            add_pil_resized = add_pil.resize((ow, oh), Image.Resampling.NEAREST)
            erase_pil_resized = erase_pil.resize((ow, oh), Image.Resampling.NEAREST)
            
            add_mask_final = np.array(add_pil_resized) > 0
            erase_mask_final = np.array(erase_pil_resized) > 0

            # 4. Áp dụng mask đã đúng kích thước
            alpha[add_mask_final] = 255
            alpha[erase_mask_final] = 0
            
        except Exception as e:
            print(f"Không có dữ liệu vẽ hoặc có lỗi: {e}")
            pass

    new_rgba = rgba.copy()
    new_rgba[..., 3] = alpha
    
    path_rgba = save_temp_png(new_rgba)
    path_mask = save_temp_png(alpha)
    
    return to_pil(new_rgba), [path_rgba, path_mask]

# ====== UI ======
# CẬP NHẬT: Thêm CSS để tùy chỉnh tiêu đề và màu nút
with gr.Blocks(theme=gr.themes.Soft(), title="ClassCut - Semantic Segmentation - Huỳnh Xuân Vỹ - Trần Duy Vương", css="""
/* Định dạng cho các tiêu đề phụ (chữ nhỏ hơn, màu xám) */
.subtitle { 
    margin-top: -15px !important; 
    margin-bottom: 10px !important; 
    color: #a0a0a0; 
    font-size: 0.9em; 
}
/* Căn giữa ảnh preview trong Sketchpad */
#mask_canvas > div {
    background-size: contain !important;
    background-repeat: no-repeat !important;
    background-position: center center !important;
}
/* Đổi màu các nút bấm từ tím sang xám đậm */
.gradio-container .gr-button {
    background: #374151 !important;
    color: white !important;
    border: none !important;
}
/* Hiệu ứng khi di chuột qua nút */
.gradio-container .gr-button:hover {
    background: #4b5563 !important;
}
""") as demo:
    gr.Markdown("## ClassCut - Semantic Segmentation ")

    state_seg = gr.State()
    state_img = gr.State()
    state_rgba_obj = gr.State()
    state_crop_info = gr.State()
    state_canvas_bg = gr.State()

    MIN_AREA_DEFAULT = 500
    FEATHER_DEFAULT = 1.0
    min_area_state = gr.State(MIN_AREA_DEFAULT)
    feather_state = gr.State(FEATHER_DEFAULT)

    with gr.Row(equal_height=False):
        # --- CỘT 1: INPUT & KẾT QUẢ SEGMENTATION ---
        with gr.Column(scale=3, min_width=360):
            # CẬP NHẬT: Dùng Markdown làm tiêu đề riêng
            gr.Markdown("### 1. Tải ảnh gốc lên")
            img_in = gr.Image(type="pil", show_label=False) # Ẩn tiêu đề cũ

            # CẬP NHẬT: Nút bấm sẽ được đổi màu bằng CSS ở trên
            btn_run = gr.Button("2. Chạy Segmentation")

            gr.Markdown("### 3. Kết quả Segmentation")
            gr.Markdown("<p class='subtitle'>Click vào ảnh để tách vật thể</p>")
            seg_vis = gr.Image(show_label=False, height=420)

        # --- CỘT 2: CHỈNH SỬA MASK & OUTPUT ---
        with gr.Column(scale=3):
            gr.Markdown("### 5. Chỉnh sửa mask")
            gr.Markdown("<p class='subtitle'>Dùng cọ TRẮNG để thêm, ĐEN để xoá</p>")
            mask_canvas = gr.Sketchpad(
                show_label=False, # Ẩn tiêu đề cũ
                height=360,
                width=540,
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                elem_id="mask_canvas"
            )
            # CẬP NHẬT: Đã xóa 'vertical_align="center"' để tương thích
            with gr.Row():
                 # CẬP NHẬT: Tách tiêu đề cho Radio button
                gr.Markdown("#### Chế độ vẽ")
                mode = gr.Radio(
                    choices=["Draw (Thêm)", "Erase (Xoá)"],
                    value="Draw (Thêm)",
                    show_label=False # Ẩn tiêu đề cũ
                )
            
            btn_apply = gr.Button("6. Áp dụng chỉnh sửa & Tạo file")

        # --- CỘT 3: OUTPUT & FILES ---
        with gr.Column(scale=3):
            info = gr.Markdown("...") # Giữ nguyên

            gr.Markdown("### 4. Vật thể đã tách (RGBA)")
            cut_out = gr.Image(show_label=False, height=320, interactive=False)

            gr.Markdown("### 7. Output sau chỉnh sửa (Preview)")
            edited_out = gr.Image(show_label=False, height=280, interactive=False)
            
            gr.Markdown("### Tệp đã xuất")
            gr.Markdown("<p class='subtitle'>[object_rgba.png, mask.png]</p>")
            files_out = gr.Files(show_label=False)

    # ==== Wiring ====
    def change_mode(m):
        color = "#FFFFFF" if "Draw" in m else "#000000"
        return gr.update(brush=gr.Brush(colors=[color], color_mode="fixed"))

    mode.change(fn=change_mode, inputs=[mode], outputs=[mask_canvas])

    btn_run.click(
        fn=run_segmentation,
        inputs=[img_in],
        outputs=[seg_vis, state_seg, state_img]
    )

    seg_vis.select(
        fn=click_extract,
        inputs=[img_in, state_seg, min_area_state, feather_state],
        outputs=[cut_out, mask_canvas, info, state_rgba_obj, state_crop_info, state_canvas_bg]
    )

    btn_apply.click(
        fn=apply_sketch,
        inputs=[state_rgba_obj, mask_canvas, state_crop_info, state_canvas_bg],
        outputs=[edited_out, files_out]
    )

# Run gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)