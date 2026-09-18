import customtkinter as ctk
from utils import fix_arabic
from image_editor import ImageEditorWindow
from video_ai_editor import VideoAIEditorWindow

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("نظام AI Vision Pro - الشاشة الرئيسية")
        self.geometry("700x450")
        self.eval('tk::PlaceWindow . center') # توسيط النافذة في الشاشة

        # العناوين
        title_font = ctk.CTkFont(size=28, weight="bold")
        ctk.CTkLabel(self, text="AI Vision Pro", font=title_font, text_color="#0284c7").pack(pady=(50, 10))
        ctk.CTkLabel(self, text=fix_arabic("اختر بيئة العمل للبدء"), font=ctk.CTkFont(size=16), text_color="gray").pack(pady=(0, 40))

        # أزرار التنقل بين النوافذ
        btn_font = ctk.CTkFont(size=15, weight="bold")
        
        self.btn_image = ctk.CTkButton(
            self, text=fix_arabic("🖼️ محرر الصور الثابتة والفلاتر"), 
            font=btn_font, height=50, width=320, 
            command=self.open_image_editor
        )
        self.btn_image.pack(pady=10)

        self.btn_video = ctk.CTkButton(
            self, text=fix_arabic("🎥 معالجة الفيديو والذكاء الاصطناعي"), 
            font=btn_font, height=50, width=320, 
            fg_color="#9333ea", hover_color="#7e22ce", 
            command=self.open_video_editor
        )
        self.btn_video.pack(pady=10)

        # التذييل
        footer = ctk.CTkLabel(
            self, 
            text=fix_arabic("إعداد: المنذر عوض | جامعة الجزيرة - قسم الذكاء الاصطناعي"), 
            text_color="gray"
        )
        footer.pack(side="bottom", pady=20)

    def open_image_editor(self):
        # يفتح ملف الصور كنافذة منبثقة مستقلة
        ImageEditorWindow(self)

    def open_video_editor(self):
        # يفتح ملف الفيديو كنافذة منبثقة مستقلة
        VideoAIEditorWindow(self)

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()