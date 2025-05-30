請先下載模型與必要檔案，並依照下方結構放入對應資料夾。

### 1️⃣ 雲端載點

🔗 [下載連結（Google Drive）](https://drive.google.com/drive/folders/1Q-gRmi4l2JtdDCf4RZIDDP-RIqlsEKFH?usp=sharing)

---

## 📁 請將下載的檔案依照以下結構放置：
Hairstyle Recommendation System/

├── pretrained_models/

│ ├── bald_proxy.pt

│ ├── ffhq.pt

│ ├── seg.pth

│ ├── sketch_proxy.pt

│ ├── shape_predictor_68_face_landmarks.dat

│ └── ffhq_PCA.npz

├── best_model.pth

├── hybrid_recommender.pth

├── recommender_model_advanced.pth


# 安裝環境
pip install -r requirements.txt

# 執行應用
python Predict_Recommendation_app.py
