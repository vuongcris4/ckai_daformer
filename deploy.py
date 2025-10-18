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
CKPT   = "work_dirs/211108_1622_gta2cs_daformer_s0_7f24c/latest.pth"

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

# ====== TIỆN ÍCH ======
def random_color_map(n=256, seed=0):
    rs = np.random.RandomState(seed)
    return (rs.rand(n,3)*255).astype(np.uint8)

COLOR_MAP = random_color_map(256, seed=0)

def to_pil(arr):
    return Image.fromarray(arr)

def save_temp_png(arr):
    """Lưu array (H,W[,C]) thành PNG tạm, trả path để DownloadButton dùng."""
    img = Image.fromarray(arr)
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path

# ====== 1) CHẠY SEGMENTATION ======
def run_segmentation(image_pil):
    img = np.array(image_pil.convert("RGB"))
    result = inference_segmentor(MODEL, img)
    seg = result[0] if isinstance(result, (list,tuple)) else result
    overlay = (0.5*img + 0.5*COLOR_MAP[seg]).astype(np.uint8)
    return to_pil(overlay), seg, img

# ====== 2) RENDER LẠI OVERLAY THEO CLASS/OPACITY ======
def render_overlay(img_np, seg, visible_classes, opacity):
    if img_np is None or seg is None:
        return None
    vis_mask = np.zeros(seg.shape, dtype=bool)
    # Nếu không chọn class nào, hiển thị tất cả
    if not visible_classes:
        visible_classes = CLASSES

    for cname in visible_classes:
        if cname in CLASSES:
            cid = CLASSES.index(cname)
            vis_mask |= (seg==cid)

    color = COLOR_MAP[seg]
    out = img_np.copy()
    out[vis_mask] = ((1-opacity)*img_np[vis_mask] + opacity*color[vis_mask]).astype(np.uint8)
    return to_pil(out)

# ====== 3) CLICK → TÁCH COMPONENT THEO ĐIỂM ======
def click_extract(evt: gr.SelectData, image_pil, seg, min_area, feather):
    if image_pil is None or seg is None or evt is None:
        return None, None, "Hãy chạy segmentation rồi click lên ảnh.", None
    x, y = evt.index
    img = np.array(image_pil.convert("RGB"))
    h, w = seg.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None, None, "Điểm click ngoài ảnh.", None

    cid = int(seg[y, x])
    cname = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"class_{cid}"
    mask = (seg == cid).astype(np.uint8)
    nlabel, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    lab_id = labels[y, x]
    area = int(stats[lab_id, cv2.CC_STAT_AREA])
    if area < min_area:
        return None, None, f"Vùng quá nhỏ (<{min_area}). Class: {cname}", None

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
    
    return output_pil, output_pil, info, rgba_crop

# ====== 4) ÁP DỤNG SỬA MASK (SKETCH) LÊN OBJECT ======
def apply_sketch(stashed_rgba_obj, sketch_data):
    """
    Áp dụng nét vẽ từ Sketchpad (Trắng=Thêm, Đen=Xóa) vào alpha channel của object.
    """
    if stashed_rgba_obj is None:
        return None, None

    rgba = stashed_rgba_obj.copy()
    alpha = rgba[..., 3] # Lấy kênh alpha (mask) gốc

    # Chỉ xử lý nếu người dùng có vẽ
    if sketch_data is not None:
        try:
            # LẤY DỮ LIỆU NÉT VẼ TỪ SKETCHPAD
            # sketch_data['composite'] là một numpy array (H, W, 4)
            composite_np = sketch_data['composite']
            
            # Chỉ lấy 3 kênh màu RGB để kiểm tra màu cọ (trắng/đen)
            sketch_rgb = composite_np[..., :3]

            # Vùng vẽ màu trắng (RGB > 200) được coi là vùng "THÊM"
            add_mask = np.all(sketch_rgb > 200, axis=-1)
            
            # Vùng vẽ màu đen (RGB < 50) được coi là vùng "XÓA"
            erase_mask = np.all(sketch_rgb < 50, axis=-1)

            # Cập nhật kênh alpha
            alpha[add_mask] = 255  # Thêm vào mask (làm cho nó đục hoàn toàn)
            alpha[erase_mask] = 0    # Xóa khỏi mask (làm cho nó trong suốt)
            
        except Exception as e:
            # Nếu có lỗi hoặc không có nét vẽ, bỏ qua
            print(f"Không có dữ liệu vẽ hoặc có lỗi: {e}")
            pass

    # Tạo ảnh RGBA mới với kênh alpha đã được chỉnh sửa
    new_rgba = rgba.copy()
    new_rgba[..., 3] = alpha
    
    # Lưu file tạm để người dùng có thể tải về
    path_rgba = save_temp_png(new_rgba)
    path_mask = save_temp_png(alpha)
    
    return to_pil(new_rgba), [path_rgba, path_mask]
    
# ====== UI ======
with gr.Blocks(theme=gr.themes.Soft(), title="ClassCut – Semantic Segmentation") as demo:
    gr.Markdown("## ClassCut – Semantic Segmentation (Giao diện tối ưu)")

    # States
    state_seg = gr.State()
    state_img = gr.State()
    state_rgba_obj = gr.State()

    # Bố cục 3 cột chính
    with gr.Row(equal_height=False):
        
        # --- CỘT 1: INPUT & CÀI ĐẶT ---
        with gr.Column(scale=2, min_width=300):
            img_in = gr.Image(type="pil", label="1. Tải ảnh gốc lên")
            btn_run = gr.Button("2. Chạy Segmentation", variant="primary")
            with gr.Accordion("Tùy chọn và bộ lọc", open=True):
                opacity = gr.Slider(0.0, 1.0, 0.55, step=0.05, label="Độ mờ của lớp phủ")
                min_area = gr.Slider(0, 20000, 500, step=50, label="Diện tích tối thiểu (lọc nhiễu)")
                feather = gr.Slider(0.0, 6.0, 1.0, step=0.5, label="Làm mềm biên (Feather σ)")

        # --- CỘT 2: KẾT QUẢ & TƯƠNG TÁC ---
        with gr.Column(scale=3):
            seg_vis = gr.Image(label="3. Kết quả Segmentation (Click vào ảnh để tách vật thể)", height=500)
            visible = gr.CheckboxGroup(CLASSES, label="Hiển thị các lớp (class)", value=CLASSES)

        # --- CỘT 3: CHỈNH SỬA & XUẤT FILE ---
        with gr.Column(scale=3):
            info = gr.Markdown("...")
            with gr.Tabs():
                with gr.TabItem("Vật thể (PNG)"):
                    cut_out = gr.Image(label="Vật thể đã tách (RGBA)", height=400, interactive=False)
                with gr.TabItem("Chỉnh sửa Mask"):
                    # Cập nhật Sketchpad để có cọ trắng/đen
                    # Sửa trong file deploy.py

                    mask_canvas = gr.Sketchpad(
                        label="4. Chỉnh sửa mask: Dùng cọ TRẮNG để thêm, cọ ĐEN để xóa",
                        height=400,
                        brush=gr.Brush(colors=["#FFFFFF", "#000000"], color_mode="fixed"),
                        # XÓA DÒNG NÀY ĐI:
                        # brush_color="#FFFFFF" 
                    )
            
            btn_apply = gr.Button("5. Áp dụng chỉnh sửa & Tạo file", variant="primary")
            files_out = gr.Files(label="Tệp đã xuất: [object_rgba.png, mask.png]")

    # ==== Wiring ====
    btn_run.click(
        fn=run_segmentation,
        inputs=[img_in],
        outputs=[seg_vis, state_seg, state_img]
    )

    opacity.change(fn=render_overlay, inputs=[state_img, state_seg, visible, opacity], outputs=[seg_vis])
    visible.change(fn=render_overlay, inputs=[state_img, state_seg, visible, opacity], outputs=[seg_vis])

    seg_vis.select(
        fn=click_extract,
        inputs=[img_in, state_seg, min_area, feather],
        outputs=[cut_out, mask_canvas, info, state_rgba_obj]
    )

    btn_apply.click(
        fn=apply_sketch,
        inputs=[state_rgba_obj, mask_canvas],
        outputs=[cut_out, files_out]
    )

# Run gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)