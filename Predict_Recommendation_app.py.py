import os
import torch
import torch.nn as nn
import numpy as np
import joblib
import random
from PIL import Image
from torchvision import transforms, models
import gradio as gr

# ==== Hair Synthesis Model ==== #
from scripts.Embedding import Embedding
from scripts.text_proxy import TextProxy
from scripts.ref_proxy import RefProxy
from scripts.sketch_proxy import SketchProxy
from scripts.bald_proxy import BaldProxy
from scripts.color_proxy import ColorProxy
from scripts.feature_blending import hairstyle_feature_blending
from utils.seg_utils import vis_seg
from utils.mask_ui import painting_mask
from utils.image_utils import display_image_list, process_display_input
from utils.model_utils import load_base_models
from utils.options import Options

# ==== Init ==== #
opts = Options().parse(jupyter=True)
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

g_ema, mean_latent_code, seg = load_base_models(opts)
ii2s = Embedding(opts, g_ema, mean_latent_code[0, 0])
bald_proxy = BaldProxy(g_ema, opts.bald_path)
text_proxy = TextProxy(opts, g_ema, seg, mean_latent_code)
ref_proxy = RefProxy(opts, g_ema, seg, ii2s)
sketch_proxy = SketchProxy(g_ema, mean_latent_code, opts.sketch_path)
color_proxy = ColorProxy(opts, g_ema, seg)

# ==== Face Shape & Hairstyle Recommendation ==== #
face_shape_classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
gender_classes = ['men', 'women']
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_model = models.resnet50(pretrained=False)
face_model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, len(face_shape_classes))
)
face_model.load_state_dict(torch.load("best_model.pth", map_location=device))
face_model.eval().to(device)

class HybridRecommender(nn.Module):
    def __init__(self, num_attrs, num_classes):
        super().__init__()
        self.cnn = models.resnet50(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.embedding = nn.Embedding(10, 8)
        self.mlp = nn.Sequential(
            nn.Linear(2048 + num_attrs * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, attr):
        img_feat = self.cnn(img)
        attr_embed = self.embedding(attr)
        attr_feat = attr_embed.view(attr.size(0), -1)
        return self.mlp(torch.cat([img_feat, attr_feat], dim=1))

label_encoder = joblib.load("label_encoder.pkl")
recommend_model = HybridRecommender(num_attrs=2, num_classes=len(label_encoder.classes_))
recommend_model.load_state_dict(torch.load("hybrid_recommender.pth", map_location=device), strict=False)
recommend_model.eval().to(device)

predicted_face_shape = ""
recommended_hairstyle = ""

def step1_predict_and_recommend(img_path, gender):
    global predicted_face_shape, recommended_hairstyle
    predicted_face_shape = predict_face_shape(img_path)
    recommended_hairstyle = recommend_hairstyle(img_path, predicted_face_shape, gender)
    return f"臉型預測：{predicted_face_shape}", f"推薦髮型：{recommended_hairstyle}"

def predict_face_shape(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_224(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(face_model(img_tensor), dim=1).item()
    return face_shape_classes[pred]

def recommend_hairstyle(img_path, face_shape, gender="women"):
    attr = torch.tensor([[face_shape_classes.index(face_shape), gender_classes.index(gender)]], dtype=torch.long).to(device)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_224(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = recommend_model(img_tensor, attr)
        hairstyle_idx = torch.topk(preds, 1).indices.item()
    return label_encoder.inverse_transform([hairstyle_idx])[0].replace('_', ' ')

def hairstyle_editing(global_cond, src_name, force_recompute=False):
    latent_path = os.path.join(opts.src_latent_dir, f"{src_name}.npz")
    img_path = os.path.join(opts.src_img_dir, f"{src_name}.jpg")

    if not os.path.isfile(latent_path) or force_recompute:
        latent_w, latent_F = ii2s.invert_image_in_FS(image_path=img_path)
        np.savez(latent_path, latent_in=latent_w.cpu().numpy(), latent_F=latent_F.cpu().numpy())

    src_latent = torch.from_numpy(np.load(latent_path)['latent_in']).cuda()
    src_feature = torch.from_numpy(np.load(latent_path)['latent_F']).cuda()
    src_image = image_transform(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
    input_mask = torch.argmax(seg(src_image)[1], dim=1).long().clone().detach()

    latent_bald, _ = bald_proxy(src_latent)
    latent_global, _ = text_proxy(global_cond, src_image, from_mean=False)

    src_feature, edited_img = hairstyle_feature_blending(
        g_ema, seg, src_latent, src_feature, input_mask,
        latent_bald=latent_bald,
        latent_global=latent_global,
        latent_local=None,
        local_blending_mask=None
    )
    return src_latent, src_feature, edited_img

def step2_generate(img_path, hairstyle_text, hairstyle_custom):
    hairstyle = hairstyle_custom.strip() or hairstyle_text or recommended_hairstyle
    src_name = os.path.splitext(os.path.basename(img_path))[0]
    src_latent, src_feature, edited_img = hairstyle_editing(hairstyle, src_name)
    _, visual_final_list = color_proxy("108157.jpg", edited_img, src_latent, src_feature)
    final_result = visual_final_list[-1]

    if isinstance(final_result, np.ndarray):
        final_result = torch.from_numpy(final_result).permute(2, 0, 1).float()
        if final_result.max() > 1.0:
            final_result /= 255.0
    else:
        final_result = final_result.detach().cpu()

    final_result = final_result.clamp(0, 1)
    if final_result.dim() == 4:
        final_result = final_result.squeeze(0)

    from torchvision.transforms.functional import to_pil_image
    result_img = to_pil_image(final_result).convert("RGB")
    return result_img

image_dir = os.path.join(opts.src_img_dir)
image_choices = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]

with gr.Blocks(title="髮型推薦系統") as demo:
    gr.Markdown("## 髮型推薦系統")
    with gr.Row():
        file_selector = gr.Dropdown(choices=image_choices, label="選擇圖片")
        gender_radio = gr.Radio(choices=gender_classes, label="選擇性別", value="women")

    step1_button = gr.Button("➤ 第一步：預測臉型與推薦髮型")
    face_shape_text = gr.Textbox(label="臉型預測")
    hairstyle_text = gr.Textbox(label="推薦髮型")

    gr.Markdown("---")
    with gr.Row():
        hairstyle_dropdown = gr.Dropdown(choices=[
            "Layered hairstyle", "Short bob", "Long curls", "Pixie cut", "Korean perm"
        ], label="快速選擇髮型")
        hairstyle_custom = gr.Textbox(label="或自訂髮型描述", placeholder="例如：Korean side bang style")

    step2_button = gr.Button("➤ 第二步：執行髮型合成")
    output_img = gr.Image(type="pil", label="合成後的結果")

    step1_button.click(fn=step1_predict_and_recommend,
                       inputs=[file_selector, gender_radio],
                       outputs=[face_shape_text, hairstyle_text])

    step2_button.click(fn=step2_generate,
                       inputs=[file_selector, hairstyle_dropdown, hairstyle_custom],
                       outputs=output_img)

demo.launch()