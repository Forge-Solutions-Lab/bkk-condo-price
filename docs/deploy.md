# Deploy หน้าเว็บขึ้น Streamlit Community Cloud

เป้าหมาย: ได้ลิงก์ `https://<ชื่อ>.streamlit.app` ที่เปิดใช้ได้จากทุกเครื่อง สำหรับหัวข้อ Live demo

## สิ่งที่ repo มีพร้อมแล้ว
| ไฟล์ | ใช้ทำอะไร |
|---|---|
| `requirements.txt` | Cloud ติดตั้ง library ตามเวอร์ชันที่ pin ไว้ |
| `app/streamlit_app.py` | ไฟล์หลักของหน้าเว็บ |
| `models/model_bundle.pkl` | โมเดลที่ฝึกแล้ว (ไม่ต้องฝึกใหม่บน Cloud) |
| `reports/forecast.json` | ตารางคาดการณ์ราคาปี 2026-2036 |
| `.streamlit/config.toml`, `app/theme.css` | ธีมดำ-ทอง |

ทดสอบแล้ว (24 พ.ย. 2569): clone repo ใหม่จาก GitHub, ติดตั้ง `requirements.txt` บน Python 3.12 แล้วรันหน้าเว็บ ได้ราคาเท่ากับในเครื่องที่ใช้พัฒนา

## ขั้นตอน (เจ้าของบัญชี GitHub ทำเอง เพราะต้องล็อกอิน)
1. เปิด https://share.streamlit.io แล้วล็อกอินด้วย GitHub
2. ถ้าถูกถามสิทธิ์ ให้อนุญาต Streamlit เข้าถึง organization `Forge-Solutions-Lab`
3. กดปุ่มสร้างแอปใหม่ (Create app) แล้วเลือกแบบ deploy จาก GitHub
4. กรอก:
   - Repository: `Forge-Solutions-Lab/bkk-condo-price`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: ตั้งชื่อเอง เช่น `bkk-condo-price`
5. เปิด Advanced settings แล้วเลือก Python เวอร์ชันใหม่ที่สุดที่มีให้เลือก (ทดสอบแล้วว่า 3.12 และ 3.14 ใช้ได้)
6. กด Deploy แล้วรอติดตั้งประมาณ 3-5 นาที

ชื่อปุ่มบนหน้าเว็บอาจต่างจากที่เขียนไว้เล็กน้อย ถ้าหาไม่เจอให้ดูคำที่ใกล้เคียง

## ตรวจว่าใช้ได้จริง
- เปิดลิงก์จากมือถือหรือเครื่องเพื่อน
- กด "ประเมินราคา" ด้วยค่าเริ่มต้น ต้องได้ประมาณ 5,743,183 บาท (โมเดลหลัก) และ 4,759,303 บาท (KNN) และมีกล่อง "ราคานี้เชื่อได้แค่ไหน"
- เลือกโครงการ 1 โครงการ แล้วดูว่ามีบรรทัด "ข้อมูลตึกที่ใช้"
- เลื่อนปีเป็น 2036 ต้องมีกราฟราคาคาดการณ์

## หลังจาก deploy
- ทุกครั้งที่ merge เข้า `main` หน้าเว็บจะอัปเดตเองภายในไม่กี่นาที
- แอปที่ไม่มีคนเข้าระยะหนึ่งจะหลับ ครั้งแรกที่เปิดอาจต้องกดปุ่มปลุกแล้วรอสักครู่ ควรเปิดทิ้งไว้ก่อนพรีเซนต์
- ใส่ลิงก์ที่ได้ใน README หัวข้อ "Live demo" ผ่าน PR ถัดไป

---

# Deploy หน้าเว็บขึ้น Vercel (อัปเดตอัตโนมัติจาก `main`)

ลิงก์ใช้งานจริง (Production): **https://bkk-condo-price.vercel.app**

## โครงสร้างไฟล์สำหรับ Vercel
| ไฟล์ | หน้าที่ |
|---|---|
| `public/index.html`, `public/styles.css`, `public/app.js` | หน้าเว็บธีมดำ-ทอง (เสิร์ฟตรงจาก Vercel CDN) |
| `api/index.py` | FastAPI backend (`/api/health`, `/api/meta`, `/api/zones`, `/api/projects`, `/api/predict`) |
| `app/inference.py` | ลอจิกประเมินราคา ความน่าเชื่อถือ และคาดการณ์ที่ใช้ร่วมกันระหว่าง Vercel และ Streamlit |
| `pyproject.toml`, `vercel.json`, `.vercelignore` | กำหนด Python 3.14 (Fluid Compute), dependency เฉพาะ runtime และ cache/security headers |

## การอัปเดตอัตโนมัติ
- โปรเจกต์ Vercel เชื่อมกับสาขา `main` ของ `Forge-Solutions-Lab/bkk-condo-price` โดยตรง
- ทุกครั้งที่ push หรือ merge PR เข้า `main` Vercel จะ build ใหม่และอัปเดต https://bkk-condo-price.vercel.app อัตโนมัติภายใน ~30-45 วินาที
- การอัปเดต `models/model_bundle.pkl`, `reports/forecast.json`, `src/` หรือ `app/inference.py` มีผลกับทั้ง Vercel และ Streamlit พร้อมกัน
- หากต้องการแก้หน้าตา UI บน Vercel ให้แก้ไฟล์ใน `public/` (ส่วน `app/streamlit_app.py` และ `app/theme.css` ใช้กับหน้า Streamlit)

## รันฝั่ง Vercel ทดสอบบนเครื่อง
```bash
pip install fastapi uvicorn
uvicorn api.index:app --reload --port 8000
```
แล้วเปิด http://localhost:8000

