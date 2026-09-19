import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image
import numpy as np
import os
import urllib.request
from utils import fix_arabic

class VideoAIEditorWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)

        self.title(fix_arabic("معالجة الفيديو والذكاء الاصطناعي اللحظي - AI Vision Pro"))
        self.geometry("1400x900")
        self.minsize(1200, 750)
        self.focus() # جعل النافذة في المقدمة

        # المتغيرات الأساسية
        self.working_image = None 
        self.current_image = None 
        self.second_image = None 
        self.last_raw_frame = None 
        
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.is_recording = False
        self.video_writer = None
        
        # متغيرات شريط الزمن
        self.total_frames = 0
        self.fps = 30
        self._is_updating_slider = False
        
        self.history = []
        self.redo_stack = []

        # تهيئة الذكاء الاصطناعي
        self.face_cascade = None
        try:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
        except:
            self.bg_subtractor = None
        self.net = None
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
            
        self.download_ai_models()
        self.setup_ui()

    def download_ai_models(self):
        proto_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel"
        haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        
        proto_path = "MobileNetSSD_deploy.prototxt"
        model_path = "MobileNetSSD_deploy.caffemodel"
        haar_path = "haarcascade_frontalface_default.xml"

        try:
            if not os.path.exists(haar_path): urllib.request.urlretrieve(haar_url, haar_path)
            self.face_cascade = cv2.CascadeClassifier(haar_path)
        except Exception as e: print("Error Face:", e)

        try:
            if not os.path.exists(proto_path): urllib.request.urlretrieve(proto_url, proto_path)
            if not os.path.exists(model_path): urllib.request.urlretrieve(model_url, model_path)
            if hasattr(cv2.dnn, 'readNetFromCaffe'): self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            else: self.net = cv2.dnn.readNet(model=model_path, config=proto_path)
        except Exception as e: print("Error Object:", e)

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ================== الشريط الجانبي ==================
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#1e1e21")
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="AI Vision Pro\nVideo Editor", font=ctk.CTkFont(size=20, weight="bold"), text_color="#9333ea").pack(pady=(20, 10))

        btn_font = ctk.CTkFont(size=13)
        ctk.CTkButton(self.sidebar, text=fix_arabic("📁 فتح فيديو"), font=btn_font, command=self.load_media).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text=fix_arabic("📷 كاميرا مباشرة"), font=btn_font, fg_color="#0284c7", command=self.start_live_camera).pack(pady=5, padx=20, fill="x")
        
        # أزرار التحكم بالعرض
        ctk.CTkButton(self.sidebar, text=fix_arabic("⏹️ إيقاف العرض نهائياً"), font=btn_font, fg_color="#b91c1c", command=self.stop_media).pack(pady=5, padx=20, fill="x")
        
        self.btn_pause = ctk.CTkButton(self.sidebar, text=fix_arabic("⏸️ إيقاف مؤقت"), font=btn_font, fg_color="#ea580c", hover_color="#c2410c", command=self.toggle_pause)
        self.btn_pause.pack(pady=5, padx=20, fill="x")
        
        self.btn_record = ctk.CTkButton(self.sidebar, text=fix_arabic("🔴 تسجيل الفيديو"), font=btn_font, fg_color="#9333ea", command=self.toggle_recording)
        self.btn_record.pack(pady=5, padx=20, fill="x")
        
        ctk.CTkButton(self.sidebar, text=fix_arabic("💾 حفظ الإطار كصورة"), font=btn_font, command=self.save_image).pack(pady=5, padx=20, fill="x")
        
        ctk.CTkLabel(self.sidebar, text=fix_arabic("التحكم أثناء الإيقاف"), font=ctk.CTkFont(size=13, weight="bold"), text_color="gray").pack(pady=(15, 5))
        ctk.CTkButton(self.sidebar, text=fix_arabic("↩️ تراجع"), fg_color="#475569", command=self.undo).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text=fix_arabic("↪️ إعادة"), fg_color="#15803d", command=self.redo).pack(pady=5, padx=20, fill="x")
        ctk.CTkButton(self.sidebar, text=fix_arabic("🔄 إعادة للأصل"), fg_color="#c2410c", command=self.reset_image).pack(pady=5, padx=20, fill="x")

        # ================== مساحة العرض ==================
        self.main_frame = ctk.CTkFrame(self, fg_color="#121212", corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1) # لملء الشاشة بالصورة
        self.main_frame.grid_rowconfigure(1, weight=0) # لشريط الزمن
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.image_display = ctk.CTkLabel(self.main_frame, text=fix_arabic("الرجاء تحميل فيديو أو فتح الكاميرا"), text_color="gray", font=ctk.CTkFont(size=18))
        self.image_display.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.image_display.bind("<Configure>", self.on_resize)

        # إضافة شريط الزمن (Timeline) أسفل العرض
        self.timeline_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.timeline_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 15))
        
        self.timeline_slider = ctk.CTkSlider(self.timeline_frame, from_=0, to=100, command=self.on_seek)
        self.timeline_slider.set(0)
        self.timeline_slider.pack(side="left", fill="x", expand=True, padx=(0, 15))
        self.timeline_slider.configure(state="disabled") # معطل افتراضياً حتى يتم تحميل الفيديو
        
        self.time_label = ctk.CTkLabel(self.timeline_frame, text="00:00 / 00:00", font=ctk.CTkFont(size=12))
        self.time_label.pack(side="right")

        # ================== لوحة الأدوات (Tabs) الشاملة ==================
        self.tools_panel = ctk.CTkTabview(self, width=350)
        self.tools_panel.grid(row=0, column=2, padx=15, pady=15, sticky="nsew")

        tabs = ["الذكاء الاصطناعي", "الألوان", "الفلاتر والحواف", "المورفولوجي", "الميزات والتقطيع", "دمج صورتين", "الأبعاد والرسم", "أدوات ذكية"]
        for tab in tabs: self.tools_panel.add(fix_arabic(tab))
        self.sliders = {}

        self.ai_var = ctk.StringVar(value="بدون ذكاء اصطناعي")
        ai_models = ["بدون ذكاء اصطناعي", "اكتشاف الوجوه (Face Detection)", "تتبع وتشويش الوجوه (Privacy)", "التعرف على الأشياء (MobileNet)", "عزل الحركة (MOG2)"]
        for m in ai_models: ctk.CTkRadioButton(self.tools_panel.tab(fix_arabic("الذكاء الاصطناعي")), text=fix_arabic(m), variable=self.ai_var, value=m).pack(pady=10, padx=10, anchor="w")

        self.setup_slider(self.tools_panel.tab(fix_arabic("الألوان")), "السطوع (Brightness)", -100, 100, 0)
        self.setup_slider(self.tools_panel.tab(fix_arabic("الألوان")), "التباين (Contrast)", 0.5, 3.0, 1.0)
        self.setup_slider(self.tools_panel.tab(fix_arabic("الألوان")), "التشبع (Saturation)", 0.0, 3.0, 1.0)
        self.setup_slider(self.tools_panel.tab(fix_arabic("الألوان")), "جاما (Gamma Power-Law)", 0.1, 3.0, 1.0)
        ctk.CTkButton(self.tools_panel.tab(fix_arabic("الألوان")), text=fix_arabic("✅ تثبيت الألوان (عند الإيقاف)"), command=self.apply_sliders_to_working).pack(pady=5, fill="x", padx=20)
        ctk.CTkButton(self.tools_panel.tab(fix_arabic("الألوان")), text=fix_arabic("تسوية المدرج (Hist. Equalization)"), fg_color="#4f46e5", command=self.apply_hist_eq).pack(pady=5, fill="x", padx=20)
        ctk.CTkButton(self.tools_panel.tab(fix_arabic("الألوان")), text=fix_arabic("تحويل لوغاريتمي (Log Transform)"), fg_color="#4f46e5", command=self.apply_log_transform).pack(pady=5, fill="x", padx=20)

        self.filters_scroll = ctk.CTkScrollableFrame(self.tools_panel.tab(fix_arabic("الفلاتر والحواف")), fg_color="transparent")
        self.filters_scroll.pack(fill="both", expand=True)
        self.effect_var = ctk.StringVar(value="بدون تأثير")
        effects = ["بدون تأثير", "تدرج رمادي", "عكس الألوان (Invert)", "تمويه (Gaussian)", "تمويه (Average)", "إزالة الضوضاء (Median)", "زيادة الحدة (Laplacian Sharpening)", "كشف الحواف (Canny)", "حواف (Sobel X+Y)", "حواف (Prewitt)", "حواف (Laplacian)", "بوصلة كيرش (Kirsch 8-Directions)", "بوصلة روبنسون (Robinson)"]
        for eff in effects: ctk.CTkRadioButton(self.filters_scroll, text=fix_arabic(eff), variable=self.effect_var, value=eff, command=self.update_preview).pack(pady=7, padx=10, anchor="w")

        morphs = [("تآكل (Erosion)", "erosion"), ("تمدد (Dilation)", "dilation"), ("فتح (Opening)", "opening"), ("إغلاق (Closing)", "closing"), ("تدرج مورفولوجي (Gradient)", "gradient"), ("قبعة علوية (Top Hat)", "tophat"), ("قبعة سوداء (Black Hat)", "blackhat")]
        for label, op in morphs: ctk.CTkButton(self.tools_panel.tab(fix_arabic("المورفولوجي")), text=fix_arabic(label), command=lambda o=op: self.apply_morphology(o)).pack(pady=5, fill="x", padx=20)

        self.feat_scroll = ctk.CTkScrollableFrame(self.tools_panel.tab(fix_arabic("الميزات والتقطيع")), fg_color="transparent")
        self.feat_scroll.pack(fill="both", expand=True)
        self.seg_var = ctk.StringVar(value="بدون تقطيع")
        seg_options = ["بدون تقطيع", "اكتشاف الخطوط (Hough Lines)", "اكتشاف الدوائر (Hough Circles)", "اكتشاف الزوايا (Shi-Tomasi)", "تقطيع (Otsu Threshold)", "تقطيع متكيف (Adaptive Mean)", "تقطيع متكيف (Adaptive Gaussian)", "عزل اللون الأحمر (Color Masking)"]
        for opt in seg_options: ctk.CTkRadioButton(self.feat_scroll, text=fix_arabic(opt), variable=self.seg_var, value=opt, command=self.update_preview).pack(pady=7, padx=10, anchor="w")

        ctk.CTkButton(self.tools_panel.tab(fix_arabic("دمج صورتين")), text=fix_arabic("📂 تحميل الصورة الثانية"), fg_color="#0284c7", command=self.load_second_image).pack(pady=10, fill="x", padx=20)
        self.lbl_second_img = ctk.CTkLabel(self.tools_panel.tab(fix_arabic("دمج صورتين")), text=fix_arabic("لم يتم تحميل صورة ثانية"), text_color="gray")
        self.lbl_second_img.pack(pady=5)
        self.blend_var = ctk.StringVar(value="بدون دمج")
        blend_ops = ["بدون دمج", "جمع (Addition)", "طرح (Subtraction)", "مزج (Blending 50%)", "فرق مطلق (Absolute Diff)", "AND منطقي", "OR منطقي", "XOR منطقي"]
        for b_op in blend_ops: ctk.CTkRadioButton(self.tools_panel.tab(fix_arabic("دمج صورتين")), text=fix_arabic(b_op), variable=self.blend_var, value=b_op, command=self.update_preview).pack(pady=5, padx=10, anchor="w")

        ctk.CTkLabel(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), text=fix_arabic("التحويل الهندسي"), font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.setup_slider(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), "إزاحة X (Translation X)", -300, 300, 0)
        self.setup_slider(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), "إزاحة Y (Translation Y)", -300, 300, 0)
        # إضافة شريط التكبير/التصغير (الزوم)
        self.setup_slider(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), "تكبير / تصغير (Zoom)", 0.2, 5.0, 1.0)
        
        ctk.CTkButton(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), text=fix_arabic("✂️ قص الإطار (Crop)"), command=self.crop_image).pack(pady=5, fill="x", padx=20)
        ctk.CTkButton(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), text=fix_arabic("🔄 تدوير 90°"), command=lambda: self.rotate_image(cv2.ROTATE_90_CLOCKWISE)).pack(pady=5, fill="x", padx=20)
        
        ctk.CTkLabel(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), text=fix_arabic("الرسم الهندسي (Overlay)"), font=ctk.CTkFont(weight="bold")).pack(pady=(15,5))
        self.draw_var = ctk.StringVar(value="بدون رسم")
        draw_ops = ["بدون رسم", "رسم خط مستقيم", "رسم مستطيل", "رسم دائرة و بيضاوي", "كتابة نص (OpenCV)"]
        for d_op in draw_ops: ctk.CTkRadioButton(self.tools_panel.tab(fix_arabic("الأبعاد والرسم")), text=fix_arabic(d_op), variable=self.draw_var, value=d_op, command=self.update_preview).pack(pady=5, padx=10, anchor="w")

        ctk.CTkButton(self.tools_panel.tab(fix_arabic("أدوات ذكية")), text=fix_arabic("🧹 إزالة عنصر غير مرغوب (Inpaint)"), command=self.remove_object).pack(pady=10, fill="x", padx=20)
        ctk.CTkButton(self.tools_panel.tab(fix_arabic("أدوات ذكية")), text=fix_arabic("👤 عزل الخلفية بالماوس (GrabCut)"), command=self.remove_background).pack(pady=10, fill="x", padx=20)

    def setup_slider(self, parent, label, min_val, max_val, default_val):
        ctk.CTkLabel(parent, text=fix_arabic(label)).pack(pady=(5, 0), anchor="w", padx=20)
        slider = ctk.CTkSlider(parent, from_=min_val, to=max_val, command=self.update_preview)
        slider.set(default_val)
        slider.pack(pady=5, fill="x", padx=20)
        self.sliders[label] = slider

    # ================== تحميل وسائط الفيديو وادارة شريط الزمن ==================
    def load_media(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if not file_path: return
        self.stop_media()
        self.cap = cv2.VideoCapture(file_path)
        
        # تهيئة معلومات شريط الزمن
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 30
        
        if self.total_frames > 0:
            self.timeline_slider.configure(to=self.total_frames, state="normal")
            self.timeline_slider.set(0)
        else:
            self.timeline_slider.configure(state="disabled")

        self.is_playing = True
        self.is_paused = False
        self.btn_pause.configure(text=fix_arabic("⏸️ إيقاف مؤقت"), fg_color="#ea580c")
        self.video_loop()

    def start_live_camera(self):
        self.stop_media()
        self.cap = cv2.VideoCapture(0)
        
        # تعطيل شريط الزمن أثناء الكاميرا
        self.total_frames = 0
        self.timeline_slider.configure(state="disabled")
        self.time_label.configure(text="Live")

        self.is_playing = True
        self.is_paused = False
        self.btn_pause.configure(text=fix_arabic("⏸️ إيقاف مؤقت"), fg_color="#ea580c")
        self.video_loop()

    # دالة التحكم بالتمرير في شريط الزمن
    def on_seek(self, value):
        if self._is_updating_slider or self.cap is None or self.total_frames <= 0: 
            return
        
        frame_idx = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        if self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                self.last_raw_frame = frame.copy()
                self.working_image = frame.copy()
                self.current_image = self.apply_processing(frame)
                self.render_image(self.current_image)
                # نرجع فريم واحد للخلف حتى لا نتخطاه عند الاستئناف
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
            cur_sec = int(frame_idx / self.fps)
            tot_sec = int(self.total_frames / self.fps)
            self.time_label.configure(text=f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")

    def load_second_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png")])
        if file_path:
            img_data = np.fromfile(file_path, dtype=np.uint8)
            self.second_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            self.lbl_second_img.configure(text=fix_arabic("تم التحميل: ") + os.path.basename(file_path), text_color="#10b981")
            self.update_preview()

    def stop_media(self):
        self.is_playing = False
        self.is_paused = False
        self.btn_pause.configure(text=fix_arabic("⏸️ إيقاف مؤقت"), fg_color="#ea580c")
        if self.cap: self.cap.release(); self.cap = None
        if self.is_recording: self.toggle_recording()

    def toggle_pause(self):
        if not self.is_playing: return
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.btn_pause.configure(text=fix_arabic("▶️ استئناف العرض"), fg_color="#15803d", hover_color="#166534")
            if self.last_raw_frame is not None:
                self.working_image = self.last_raw_frame.copy()
                self.history = [self.working_image.copy()]
                self.redo_stack.clear()
                self.update_preview()
        else:
            self.btn_pause.configure(text=fix_arabic("⏸️ إيقاف مؤقت"), fg_color="#ea580c", hover_color="#c2410c")

    def video_loop(self):
        if self.is_playing and self.cap is not None and self.cap.isOpened():
            if not self.is_paused:
                ret, frame = self.cap.read()
                if ret:
                    self.last_raw_frame = frame.copy()
                    self.working_image = frame.copy()
                    self.current_image = self.apply_processing(frame)
                    self.render_image(self.current_image)
                    if self.is_recording and self.video_writer: self.video_writer.write(self.current_image)
                    
                    # تحديث شريط الزمن وعرض الوقت بشكل متزامن
                    if self.total_frames > 0:
                        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self._is_updating_slider = True
                        self.timeline_slider.set(current_frame)
                        self._is_updating_slider = False
                        
                        cur_sec = int(current_frame / self.fps)
                        tot_sec = int(self.total_frames / self.fps)
                        self.time_label.configure(text=f"{cur_sec//60:02d}:{cur_sec%60:02d} / {tot_sec//60:02d}:{tot_sec%60:02d}")
                else: 
                    self.stop_media()
                    return
            self.after(30, self.video_loop)

    def update_preview(self, _=None):
        if self.working_image is None: return
        if self.is_playing and not self.is_paused: return
        self.current_image = self.apply_processing(self.working_image)
        self.render_image(self.current_image)

    # ================== محرك المعالجة الشامل (يُطبق على كل فريم) ==================
    def apply_processing(self, frame):
        img = frame.copy()
        h, w = img.shape[:2]
        
        # --- إضافة التكبير والتصغير (Zoom) ---
        zoom_val = self.sliders.get("تكبير / تصغير (Zoom)", None)
        if zoom_val:
            z = zoom_val.get()
            if z != 1.0:
                if z > 1.0: # التكبير (Zoom In - Crop)
                    center_x, center_y = w / 2, h / 2
                    radius_x, radius_y = w / (2 * z), h / (2 * z)
                    min_x, max_x = max(0, int(center_x - radius_x)), min(w, int(center_x + radius_x))
                    min_y, max_y = max(0, int(center_y - radius_y)), min(h, int(center_y + radius_y))
                    
                    if max_x > min_x and max_y > min_y:
                        img = cv2.resize(img[min_y:max_y, min_x:max_x], (w, h))
                else: # التصغير (Zoom Out - Pad)
                    new_w, new_h = max(1, int(w * z)), max(1, int(h * z))
                    scaled_img = cv2.resize(img, (new_w, new_h))
                    
                    # إنشاء خلفية سوداء بحجم الصورة الأصلية
                    canvas = np.zeros_like(img)
                    x_offset = (w - new_w) // 2
                    y_offset = (h - new_h) // 2
                    
                    # وضع الصورة المصغرة في المنتصف
                    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled_img
                    img = canvas
        # -------------------------------------

        tx = self.sliders.get("إزاحة X (Translation X)", 0).get() if "إزاحة X (Translation X)" in self.sliders else 0
        ty = self.sliders.get("إزاحة Y (Translation Y)", 0).get() if "إزاحة Y (Translation Y)" in self.sliders else 0
        if tx != 0 or ty != 0:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h))

        alpha = self.sliders["التباين (Contrast)"].get()
        beta = self.sliders["السطوع (Brightness)"].get()
        if alpha != 1.0 or beta != 0: img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        sat = self.sliders["التشبع (Saturation)"].get()
        if sat != 1.0:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        gamma = self.sliders["جاما (Gamma Power-Law)"].get()
        if gamma != 1.0:
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img = cv2.LUT(img, table)

        blend_mode = self.blend_var.get()
        if blend_mode != "بدون دمج" and self.second_image is not None:
            img2_resized = cv2.resize(self.second_image, (w, h))
            if blend_mode == "جمع (Addition)": img = cv2.add(img, img2_resized)
            elif blend_mode == "طرح (Subtraction)": img = cv2.subtract(img, img2_resized)
            elif blend_mode == "مزج (Blending 50%)": img = cv2.addWeighted(img, 0.5, img2_resized, 0.5, 0)
            elif blend_mode == "فرق مطلق (Absolute Diff)": img = cv2.absdiff(img, img2_resized)
            elif blend_mode == "AND منطقي": img = cv2.bitwise_and(img, img2_resized)
            elif blend_mode == "OR منطقي": img = cv2.bitwise_or(img, img2_resized)
            elif blend_mode == "XOR منطقي": img = cv2.bitwise_xor(img, img2_resized)

        draw_mode = self.draw_var.get()
        if draw_mode != "بدون رسم":
            if draw_mode == "رسم خط مستقيم": cv2.line(img, (int(w*0.1), int(h*0.1)), (int(w*0.9), int(h*0.9)), (0, 0, 255), 4)
            elif draw_mode == "رسم مستطيل": cv2.rectangle(img, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (0, 255, 0), 3)
            elif draw_mode == "رسم دائرة و بيضاوي":
                cv2.circle(img, (int(w*0.5), int(h*0.5)), int(h*0.2), (255, 255, 255), 3)
                cv2.ellipse(img, (int(w*0.5), int(h*0.5)), (int(w*0.3), int(h*0.1)), 0, 0, 360, (255, 255, 0), 2)
            elif draw_mode == "كتابة نص (OpenCV)": cv2.putText(img, "AI Vision Pro", (int(w*0.1), int(h*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        eff = self.effect_var.get()
        if eff != "بدون تأثير":
            if eff == "تدرج رمادي": img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif eff == "عكس الألوان (Invert)": img = cv2.bitwise_not(img)
            elif eff == "تمويه (Gaussian)": img = cv2.GaussianBlur(img, (15, 15), 0)
            elif eff == "تمويه (Average)": img = cv2.blur(img, (15, 15))
            elif eff == "إزالة الضوضاء (Median)": img = cv2.medianBlur(img, 7)
            elif eff == "زيادة الحدة (Laplacian Sharpening)":
                gray_s = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                lap = cv2.Laplacian(gray_s, cv2.CV_64F, ksize=3)
                sharp = gray_s.astype(np.float64) - (0.5 * lap)
                sharp = np.clip(sharp, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR) if len(img.shape)==3 else sharp
            elif eff == "كشف الحواف (Canny)": img = cv2.Canny(img, 100, 200)
            elif eff == "حواف (Sobel X+Y)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                sx = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
                sy = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
                img = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
            elif eff == "حواف (Prewitt)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                kx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                px = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_64F, kx))
                py = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_64F, ky))
                img = cv2.addWeighted(px, 0.5, py, 0.5, 0)
            elif eff == "حواف (Laplacian)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                img = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
            elif eff == "بوصلة كيرش (Kirsch 8-Directions)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                kirsch_filters = [np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]]), np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]]),
                                  np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]]), np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]]),
                                  np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]]), np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]]),
                                  np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]]), np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])]
                edges = [cv2.filter2D(gray, cv2.CV_64F, k) for k in kirsch_filters]
                img = cv2.convertScaleAbs(np.max(edges, axis=0))
            elif eff == "بوصلة روبنسون (Robinson)":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                robinson_filters = [np.array([[1,2,1],[0,0,0],[-1,-2,-1]]), np.array([[0,1,2],[-1,0,1],[-2,-1,0]]),
                                    np.array([[-1,0,1],[-2,0,2],[-1,0,1]]), np.array([[-2,-1,0],[-1,0,1],[0,1,2]]),
                                    np.array([[-1,-2,-1],[0,0,0],[1,2,1]]), np.array([[0,-1,-2],[1,0,-1],[2,1,0]]),
                                    np.array([[1,0,-1],[2,0,-2],[1,0,-1]]), np.array([[2,1,0],[1,0,-1],[0,-1,-2]])]
                edges = [cv2.filter2D(gray, cv2.CV_64F, k) for k in robinson_filters]
                img = cv2.convertScaleAbs(np.max(edges, axis=0))

        seg_mode = self.seg_var.get()
        if seg_mode != "بدون تقطيع":
            gray_seg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            if seg_mode == "اكتشاف الخطوط (Hough Lines)":
                edges_h = cv2.Canny(gray_seg, 50, 150)
                lines = cv2.HoughLinesP(edges_h, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
                if lines is not None:
                    for line in lines: x1, y1, x2, y2 = line[0]; cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif seg_mode == "اكتشاف الدوائر (Hough Circles)":
                blur_c = cv2.medianBlur(gray_seg, 5)
                circles = cv2.HoughCircles(blur_c, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=10, maxRadius=100)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]: cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2); cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            elif seg_mode == "اكتشاف الزوايا (Shi-Tomasi)":
                corners = cv2.goodFeaturesToTrack(gray_seg, 100, 0.01, 10)
                if corners is not None:
                    corners = np.intp(corners)
                    for c in corners: x, y = c.ravel(); cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
            elif seg_mode == "تقطيع (Otsu Threshold)":
                _, img = cv2.threshold(gray_seg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif seg_mode == "تقطيع متكيف (Adaptive Mean)":
                img = cv2.adaptiveThreshold(gray_seg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            elif seg_mode == "تقطيع متكيف (Adaptive Gaussian)":
                img = cv2.adaptiveThreshold(gray_seg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            elif seg_mode == "عزل اللون الأحمر (Color Masking)":
                if len(img.shape) == 3:
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(hsv_img, np.array([0, 120, 70]), np.array([10, 255, 255]))
                    mask2 = cv2.inRange(hsv_img, np.array([170, 120, 70]), np.array([180, 255, 255]))
                    img = cv2.bitwise_and(img, img, mask=(mask1 + mask2))

        ai_mode = self.ai_var.get()
        if ai_mode != "بدون ذكاء اصطناعي":
            if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            gray_for_ai = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if ai_mode == "اكتشاف الوجوه (Face Detection)" and self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray_for_ai, 1.1, 5, minSize=(30, 30))
                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)
            elif ai_mode == "تتبع وتشويش الوجوه (Privacy)" and self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray_for_ai, 1.1, 5, minSize=(30, 30))
                for (fx, fy, fw, fh) in faces:
                    face_roi = img[fy:fy+fh, fx:fx+fw]
                    blurred_face = cv2.resize(cv2.resize(face_roi, (15, 15), interpolation=cv2.INTER_LINEAR), (fw, fh), interpolation=cv2.INTER_NEAREST)
                    img[fy:fy+fh, fx:fx+fw] = blurred_face
            elif ai_mode == "التعرف على الأشياء (MobileNet)" and self.net is not None:
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()
                for i in range(np.shape(detections)[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.6: 
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 165, 255), 2)
                        cv2.putText(img, f"{self.CLASSES[idx]}: {confidence*100:.1f}%", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif ai_mode == "عزل الحركة (MOG2)" and self.bg_subtractor is not None:
                fg_mask = self.bg_subtractor.apply(img)
                img = cv2.bitwise_and(img, img, mask=fg_mask)

        return img

    # ================== بقية الدوال المساعدة الخاصة بوضع الإيقاف ==================
    def save_state(self):
        if self.working_image is not None and (not self.is_playing or self.is_paused):
            self.history.append(self.working_image.copy())
            self.redo_stack.clear()

    def apply_sliders_to_working(self):
        if self.is_playing and not self.is_paused: return
        self.working_image = self.current_image.copy()
        self.save_state()
        self.update_preview()

    def apply_log_transform(self):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        c = 255 / np.log(1 + np.max(self.working_image))
        log_img = c * (np.log(self.working_image.astype(np.float32) + 1))
        self.working_image = np.uint8(log_img)
        self.save_state()
        self.update_preview()

    def apply_hist_eq(self):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        img_yuv = cv2.cvtColor(self.working_image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        self.working_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        self.save_state()
        self.update_preview()

    def apply_morphology(self, op_type):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        kernel = np.ones((5,5), np.uint8)
        if op_type == "erosion": self.working_image = cv2.erode(self.working_image, kernel, iterations=1)
        elif op_type == "dilation": self.working_image = cv2.dilate(self.working_image, kernel, iterations=1)
        elif op_type == "opening": self.working_image = cv2.morphologyEx(self.working_image, cv2.MORPH_OPEN, kernel)
        elif op_type == "closing": self.working_image = cv2.morphologyEx(self.working_image, cv2.MORPH_CLOSE, kernel)
        elif op_type == "gradient": self.working_image = cv2.morphologyEx(self.working_image, cv2.MORPH_GRADIENT, kernel)
        elif op_type == "tophat": self.working_image = cv2.morphologyEx(self.working_image, cv2.MORPH_TOPHAT, kernel)
        elif op_type == "blackhat": self.working_image = cv2.morphologyEx(self.working_image, cv2.MORPH_BLACKHAT, kernel)
        self.save_state()
        self.update_preview()

    def rotate_image(self, code):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        self.working_image = cv2.rotate(self.working_image, code)
        self.save_state()
        self.update_preview()

    def crop_image(self):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        roi = cv2.selectROI("Select Area", self.working_image, False)
        cv2.destroyWindow("Select Area")
        if roi != (0,0,0,0):
            x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            self.working_image = self.working_image[y:y+h, x:x+w]
            self.save_state()
            self.update_preview()

    def remove_object(self):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        roi = cv2.selectROI("Select Object", self.working_image, False)
        cv2.destroyWindow("Select Object")
        if roi != (0,0,0,0):
            x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            mask = np.zeros(self.working_image.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 255
            self.working_image = cv2.inpaint(self.working_image, mask, 3, cv2.INPAINT_TELEA)
            self.save_state()
            self.update_preview()

    def remove_background(self):
        if (self.is_playing and not self.is_paused) or self.working_image is None: return
        roi = cv2.selectROI("Select Foreground", self.working_image, False)
        cv2.destroyWindow("Select Foreground")
        if roi != (0,0,0,0):
            mask = np.zeros(self.working_image.shape[:2], np.uint8)
            bgdModel = np.zeros((1,65), np.float64)
            fgdModel = np.zeros((1,65), np.float64)
            cv2.grabCut(self.working_image, mask, roi, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            self.working_image = self.working_image * mask2[:, :, np.newaxis]
            self.save_state()
            self.update_preview()

    def toggle_recording(self):
        if not self.is_playing: return
        if not self.is_recording:
            file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
            if file_path:
                h, w = self.current_image.shape[:2]
                self.video_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
                self.is_recording = True
                self.btn_record.configure(text=fix_arabic("⏹️ إيقاف التسجيل"), fg_color="#dc2626")
        else:
            self.is_recording = False
            self.video_writer.release()
            self.video_writer = None
            self.btn_record.configure(text=fix_arabic("🔴 تسجيل الفيديو"), fg_color="#9333ea")
            messagebox.showinfo(fix_arabic("نجاح"), fix_arabic("تم حفظ الفيديو بنجاح!"))

    def save_image(self):
        if self.current_image is None: return
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if file_path:
            ext = "." + file_path.split('.')[-1]
            is_success, im_buf_arr = cv2.imencode(ext, self.current_image)
            if is_success: im_buf_arr.tofile(file_path)

    def undo(self):
        if (self.is_playing and not self.is_paused) or len(self.history) <= 1: return
        self.redo_stack.append(self.history.pop())
        self.working_image = self.history[-1].copy()
        self.update_preview()

    def redo(self):
        if (self.is_playing and not self.is_paused) or not self.redo_stack: return
        state = self.redo_stack.pop()
        self.history.append(state.copy())
        self.working_image = state.copy()
        self.update_preview()

    def reset_image(self):
        if self.is_paused and self.last_raw_frame is not None:
            self.working_image = self.last_raw_frame.copy()
            self.history = [self.working_image.copy()]
            self.redo_stack.clear()

        if "السطوع (Brightness)" in self.sliders: self.sliders["السطوع (Brightness)"].set(0)
        if "التباين (Contrast)" in self.sliders: self.sliders["التباين (Contrast)"].set(1.0)
        if "التشبع (Saturation)" in self.sliders: self.sliders["التشبع (Saturation)"].set(1.0)
        if "جاما (Gamma Power-Law)" in self.sliders: self.sliders["جاما (Gamma Power-Law)"].set(1.0)
        if "إزاحة X (Translation X)" in self.sliders: self.sliders["إزاحة X (Translation X)"].set(0)
        if "إزاحة Y (Translation Y)" in self.sliders: self.sliders["إزاحة Y (Translation Y)"].set(0)
        if "تكبير / تصغير (Zoom)" in self.sliders: self.sliders["تكبير / تصغير (Zoom)"].set(1.0) # إعادة الزوم للوضع الطبيعي

        self.effect_var.set("بدون تأثير")
        self.ai_var.set("بدون ذكاء اصطناعي")
        self.seg_var.set("بدون تقطيع")
        self.blend_var.set("بدون دمج")
        self.draw_var.set("بدون رسم")
        
        if not self.is_playing or self.is_paused:
            self.update_preview()

    def show_original_temp(self, event):
        if self.last_raw_frame is not None and self.is_paused: self.render_image(self.last_raw_frame)
    def show_current_temp(self, event):
        if self.current_image is not None and self.is_paused: self.render_image(self.current_image)
    def on_resize(self, event):
        if not self.is_playing and self.current_image is not None: self.after(100, lambda: self.render_image(self.current_image))

    def render_image(self, img_array):
        if img_array is None: return
        if len(img_array.shape) == 2: img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4: img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGB)
        else: img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
        pil_image = Image.fromarray(img_rgb)
        lbl_w, lbl_h = self.image_display.winfo_width(), self.image_display.winfo_height()
        if lbl_w <= 1: lbl_w, lbl_h = 800, 600

        pil_image.thumbnail((lbl_w, lbl_h), Image.Resampling.LANCZOS)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(pil_image.width, pil_image.height))
        self.image_display.configure(image=ctk_image, text="")
        self.image_display.image = ctk_image
