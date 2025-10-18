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
    result = inference_segmentor(MODEL, img)    # API MMSeg reisze ảnh đúng kích thước model, đưa qua model, nhận đầu ra là mask phân lớp cho từng pixel
    seg = result[0] if isinstance(result, (list,tuple)) else result
    overlay = (0.5*img + 0.5*COLOR_MAP[seg]).astype(np.uint8)   # HxW seg là mảng index, thay các color (3 channels) vào mảng seg
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
    if img_np is None or seg is None:   # ảnh gốc (H, W, 3)
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
        return None, None, "Hãy chạy segmentation rồi click lên ảnh.", None
    x, y = evt.index    # toạ độ x, y được clicked
    img = np.array(image_pil.convert("RGB"))
    h, w = seg.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None, None, "Điểm click ngoài ảnh.", None

    cid = int(seg[y, x])    # Lấy ID class pixel (x,y)
    cname = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"class_{cid}" # Tra tên class
    mask = (seg == cid).astype(np.uint8)    # Mask = 1 tại pixel thuộc class đó
    """
    cv2.connectedComponentsWithStats() chia mask thành các vùng liên thông (connected components):
    nlabel: số lượng vùng
    labels: ảnh gán nhãn vùng (mỗi vùng một số ID)
    stats: thống kê vùng (tọa độ, kích thước, diện tích)
    _: tâm vùng (không dùng ở đây)
    """
    nlabel, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)    
    lab_id = labels[y, x]   # labels[y, x] cho biết pixel click thuộc vùng liên thông nào (lab_id).
    area = int(stats[lab_id, cv2.CC_STAT_AREA]) # Lấy diện tích vùng đó (area).
    if area < min_area: # → Nếu object quá nhỏ thì bỏ qua (tránh click nhầm vào noise).
        return None, None, f"Vùng quá nhỏ (<{min_area}). Class: {cname}", None

    comp = (labels == lab_id).astype(np.uint8)  # comp là mask nhị phân chỉ vùng được click.
    if feather > 0: # Làm mềm biên (feathering) bằng Gaussian Blur — tạo vùng alpha mượt mà khi crop ra.
        comp = cv2.GaussianBlur(comp.astype(np.float32), (0,0), feather)    
        comp = np.clip(comp, 0, 1)

    alpha = (comp*255).astype(np.uint8) # comp (0–1) → nhân 255 → alpha 8-bit.
    rgba = np.dstack([img, alpha])  # Ghép img (RGB) với alpha thành ảnh RGBA (4 kênh).
    x0,y0,wc,hc = stats[lab_id,0], stats[lab_id,1], stats[lab_id,2], stats[lab_id,3]    # Lấy bounding box của component từ stats.
    rgba_crop = rgba[y0:y0+hc, x0:x0+wc]    # Crop ảnh RGBA đúng vùng đó.
    output_pil = to_pil(rgba_crop)  # Chuyển ảnh RGBA crop sang PIL.Image để hiển thị/tải.
    info = f"Class: {cname} (id={cid}) • Area: {area} px • BBox: {wc}×{hc}" # Tạo chuỗi thông tin mô tả đối tượng.
    
    """
    Thường Gradio dùng 4 output:
    Ảnh hiển thị preview
    Ảnh để tải về
    Chuỗi thông tin text
    Mảng RGBA gốc (để xử lý thêm nếu cần)
    """
    """
    output_pil, được gán cho cut_out
    output_pil, được gán cho mask_canvas

    """
    return output_pil, output_pil, info, rgba_crop

# ====== 4) ÁP DỤNG SỬA MASK (SKETCH) LÊN OBJECT ======
def apply_sketch(stashed_rgba_obj, sketch_data):
    """
    Áp dụng nét vẽ từ Sketchpad (Trắng=Thêm, Đen=Xóa) vào alpha channel của object.
    """
    if stashed_rgba_obj is None:
        return None, None

    rgba = stashed_rgba_obj.copy()  # Lấy bản sao RGBA (để không phá dữ liệu gốc).
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
# (Refactor theo yêu cầu: bỏ tuỳ chọn bộ lọc, bỏ bảng chọn class, 
# chuyển "Kết quả Segmentation" sang Cột 1; Cột 2 hiển thị object + chế độ vẽ; 
# Cột 3 hiển thị khu vực chỉnh sửa mask và OUTPUT sau chỉnh sửa)
with gr.Blocks(theme=gr.themes.Soft(), title="ClassCut - Semantic Segmentation", css="""
#maskpad canvas { width: 100% !important; height: 100% !important; object-fit: contain; }
""") as demo:
    gr.Markdown("## ClassCut - Semantic Segmentation ")

    #  gr.State() dùng để lưu dữ liệu tạm giữa các bước tương tác (không hiển thị ra ngoài).
    state_seg = gr.State()  # ảnh gốc (numpy)
    state_img = gr.State()  # kết quả segmentation (mask 1 kênh)
    state_rgba_obj = gr.State() # object RGBA sau khi tách (để truyền qua Sketchpad)

    # Ẩn cấu hình lọc bằng state mặc định (theo yêu cầu bỏ UI):
    MIN_AREA_DEFAULT = 500
    FEATHER_DEFAULT = 1.0
    min_area_state = gr.State(MIN_AREA_DEFAULT)
    feather_state = gr.State(FEATHER_DEFAULT)

    """
    Bố cục 3 cột:
    - Cột 1: Input + nút chạy + Kết quả Segmentation (click để tách)
    - Cột 2: Vật thể đã tách (RGBA) + Output sau chỉnh sửa (Preview) + Files
    - Cột 3: Khu vực chỉnh sửa mask (Sketchpad) + Chế độ vẽ + nút Áp dụng
    """
    with gr.Row(equal_height=False):
        # --- CỘT 1: INPUT & KẾT QUẢ SEGMENTATION ---
        with gr.Column(scale=3, min_width=360):
            img_in = gr.Image(type="pil", label="1. Tải ảnh gốc lên")
            btn_run = gr.Button("2. Chạy Segmentation", variant="primary")
            seg_vis = gr.Image(label="3. Kết quả Segmentation (Click vào ảnh để tách vật thể)", height=420)

        # --- CỘT 2: CHỈNH SỬA MASK & OUTPUT ---
        with gr.Column(scale=3):
            mask_canvas = gr.Sketchpad(
                label="5. Chỉnh sửa mask: Dùng cọ TRẮNG để thêm, ĐEN để xoá",
                height=360,
                brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),  # mặc định TRẮNG = Draw
            )
            with gr.Row():
                mode = gr.Radio(
                    choices=["Draw (Thêm)", "Erase (Xoá)"],
                    value="Draw (Thêm)",
                    label="Chế độ vẽ"
                )
            btn_apply = gr.Button("6. Áp dụng chỉnh sửa & Tạo file", variant="primary")

        with gr.Column(scale=3):
            info = gr.Markdown("...")
            cut_out = gr.Image(label="4. Vật thể đã tách (RGBA)", height=320, interactive=False)
            edited_out = gr.Image(label="7. Output sau chỉnh sửa (Preview)", height=280, interactive=False)
            files_out = gr.Files(label="Tệp đã xuất: [object_rgba.png, mask.png]")

    # ==== Wiring ====
    # Đổi màu cọ theo Radio (Draw=trắng, Erase=đen)
    def change_mode(m):
        color = "#FFFFFF" if "Draw" in m else "#000000"
        return gr.update(brush=gr.Brush(colors=[color], color_mode="fixed"))

    mode.change(fn=change_mode, inputs=[mode], outputs=[mask_canvas])

    """ Khi bấm chạy Segmentation:
    Gọi run_segmentation(image_pil)
    Trả về:
        seg_vis: ảnh segmentation hiển thị
        state_seg: lưu mask segmentation
        state_img: lưu ảnh gốc để dùng sau
    """
    btn_run.click(
        fn=run_segmentation,
        inputs=[img_in],
        outputs=[seg_vis, state_seg, state_img]
    )

    """
    Khi click vào ảnh segmentation → gọi click_extract:
        Tách object tại vị trí click
        Hiển thị ảnh RGBA đã tách (cut_out)
        Đưa ảnh đó vào Sketchpad (mask_canvas)
        Lưu state_rgba_obj để dùng ở bước chỉnh sửa
    """
    seg_vis.select(
        fn=click_extract,
        inputs=[img_in, state_seg, min_area_state, feather_state],
        outputs=[cut_out, mask_canvas, info, state_rgba_obj]
    )

    """
    Khi nhấn “Áp dụng chỉnh sửa & Tạo file”:
        Lấy dữ liệu vẽ từ mask_canvas
        Cập nhật alpha mask object bằng apply_sketch
        Cập nhật preview OUTPUT sau chỉnh sửa (edited_out) và hiển thị 2 file tải xuống (files_out)
    """
    btn_apply.click(
        fn=apply_sketch,
        inputs=[state_rgba_obj, mask_canvas],
        outputs=[edited_out, files_out]
    )

# Run gradio
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
