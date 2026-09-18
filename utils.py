import arabic_reshaper
from bidi.algorithm import get_display

def fix_arabic(text):
    # دالة لتصحيح شكل واتجاه النص العربي ليطابق الواجهات
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    except:
        return text