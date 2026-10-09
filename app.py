import re
import joblib
import streamlit as st
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl"), joblib.load("vectorizer.pkl")

model, vectorizer = load_model()

# ต้องเหมือนกับ clean_text ใน Colab ทุกตัวอักษร
stopwords = set(thai_stopwords()) - {"ไม่", "ไม่ได้", "ไม่มี", "ดี", "มาก", "ชอบ"}

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^฀-๿a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    words = word_tokenize(text, engine="newmm", keep_whitespace=False)
    return " ".join(w for w in words if w not in stopwords)

st.title("💬 วิเคราะห์ความรู้สึกจากข้อความ")
st.write("พิมพ์ความคิดเห็นของลูกค้า แล้วระบบจะบอกว่าเป็นเชิงบวกหรือเชิงลบ")

text = st.text_area("ความคิดเห็น", placeholder="เช่น อาหารอร่อยมาก บริการดี")

if st.button("วิเคราะห์", type="primary"):
    cleaned = clean_text(text)
    if not cleaned:
        st.warning("กรุณาพิมพ์ข้อความก่อน")
    else:
        vec = vectorizer.transform([cleaned])
        label = model.predict(vec)[0]
        score = abs(model.decision_function(vec)[0])
        if label == "positive":
            st.success(f"😊 เชิงบวก (Positive) — คะแนน {score:.2f}")
        else:
            st.error(f"😞 เชิงลบ (Negative) — คะแนน {score:.2f}")
        with st.expander("ดูข้อความหลังตัดคำ"):
            st.write(cleaned)
