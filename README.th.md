# forest_ai

สกัดข้อมูลต้นไม้รายต้นจาก point cloud ของป่า — ป้อนไฟล์ `.las`/`.laz` เข้าไป
โปรแกรมจะจำแนกพื้นดิน สร้างแบบจำลองภูมิประเทศ หาลำต้นทุกต้น แยก point cloud
ออกเป็นต้น ๆ แล้ววัด **DBH ความสูง และเรือนยอด** ของแต่ละต้น
พร้อมติดธงคุณภาพกำกับทุกค่าที่วัดได้

เขียนด้วย NumPy/SciPy/scikit-learn ล้วน ไม่ต้องใช้ GPU ไม่ต้องใช้ PDAL/Open3D ไม่มี build step

*[Read in English →](README.md)*

---

## เริ่มใช้งาน

```bash
pip install -r requirements.txt

python launch.py          # เปิดเซิร์ฟเวอร์ (ถ้ายังไม่เปิด) แล้วเปิดหน้าเว็บให้เลย
python run_pipeline.py    # หรือใช้บรรทัดคำสั่ง ผลลัพธ์ลง outputs/
```

บน Windows **ดับเบิลคลิก `forest_ai.bat`** ได้ผลเหมือนกัน และ
`python tools/make_shortcuts.py` จะสร้างช็อตคัตพร้อมไอคอนไว้ที่ Desktop และ Start Menu
ส่วน `python serve.py` ยังใช้ได้ถ้าอยากเห็นหน้าต่าง console ไว้ดู log

ทั้งสองทางเรียก `forest_ai/pipeline.py` ตัวเดียวกัน ตัวเลขจึงตรงกันเสมอ
รันครั้งแรกบน 15.8 ล้านจุดใช้เวลา ~90 วินาทีบน 8 cores รันซ้ำใช้ DTM จาก cache เหลือ ~12 วินาที

> **หมายเหตุสำหรับ Windows:** ถ้า `AppData\Roaming\Python\Python3xx\Scripts`
> ไม่ได้อยู่ใน PATH ให้เรียกผ่าน `python -m ...` และต้อง `cd` เข้าโฟลเดอร์โปรเจคก่อนรัน
> เพราะโปรแกรมหาไฟล์ `.las` จาก working directory และเขียน `.cache/` กับ `outputs/` ลงที่นั่น

## ข้อมูลที่ใช้ได้

- **ต้องเป็นการสแกนจากภาคพื้นดิน** — TLS, MLS หรือ photogrammetry จากระดับพื้น
  ต้องมองเห็นลำต้นที่ระดับอก
- **ใช้ข้อมูลจากโดรนไม่ได้** — มองจากด้านบนเห็นแต่เรือนยอด แทบไม่เห็นลำต้น
  จึงวัด DBH ไม่ได้เลย
- ความหนาแน่นพอที่จะเห็นหน้าตัดลำต้น ประมาณ 100 จุด/ตร.ม. ขึ้นไป

RGB, intensity, return number มีก็ดี ไม่มีก็ได้ — pipeline ทำงานจากรูปทรงล้วน ๆ

---

## ผลลัพธ์ที่ได้

| ไฟล์ | เนื้อหา |
|---|---|
| `trees.csv` | หนึ่งแถวต่อหนึ่งต้น: ตำแหน่ง, DBH, ความสูง, เรือนยอด, ธงคุณภาพ |
| `tree_features.csv` | 51 features เชิงเรขาคณิตต่อต้น พร้อมสำหรับ train ตัวจำแนกชนิด |
| `Forest_segmented.las` | point cloud ที่ติด scalar field `tree_id` และ `height_norm` |
| `qc_01…05.png` | แบบจำลองพื้น, แผนผังลำต้น, การกระจาย, cross-section รายต้น, การ fit DBH |

คอลัมน์สำคัญใน `trees.csv`:

```
tree_id  x  y                ตำแหน่งโคนต้น (พิกัดเดียวกับ point cloud)
dbh_cm                       เส้นผ่านศูนย์กลางที่ 1.3 m
height_m                     ความสูง (percentile 99.5 กันจุด noise)
crown_base_m  crown_diameter_m  crown_area_m2  crown_volume_m3
basal_area_m2  h_d_ratio  stem_lean_deg
quality                      good / fair / poor   <-- กรองด้วยตัวนี้เสมอ
dbh_arc                      0-1 สแกนเห็นลำต้นรอบด้านแค่ไหน
dbh_rmse_cm                  residual ของการ fit วงกลม
dbh_vs_stack                 ความสอดคล้องของการวัด 2 วิธีที่อิสระต่อกัน
n_stem_layers                จำนวนชั้นที่ต่อกันได้ (4-10)
dist_from_scan_centre_m      ตัวทำนายคุณภาพที่แม่นที่สุด
structural_group             กลุ่มจาก K-means — ยังไม่ใช่ชนิดต้นไม้
```

**ถ้าต้องการตัวเลขที่เชื่อถือได้ ให้กรอง `quality == "good"` เสมอ**
บนชุดข้อมูลที่ใช้พัฒนา ค่า correlation ระหว่างความสูงกับเส้นผ่านศูนย์กลาง
ใช้ทุกต้นได้ **-0.12** (ไม่มีความหมาย) แต่ใช้เฉพาะต้น good ได้ **+0.45**
(ถูกต้องตามหลักชีววิทยา) — ส่วนต่างนี้คือหลักฐานว่าธงคุณภาพทำงานจริง

---

## หลักการทำงาน

**1 · อ่านและลบ noise** — อ่านเฉพาะ X/Y/Z ทีละ chunk เป็น float32 (15.8 ล้านจุด = 190 MB)
ลบจุดโดดเดี่ยวด้วย voxel-count filter แทน KD-tree statistical filter
ได้ผลเท่ากันกับจุด noise แต่เร็วกว่าหลายสิบเท่า

**2 · จำแนกพื้น สร้าง DTM และ normalize ความสูง** — เขียน **CSF (Cloth Simulation Filter)**
ใหม่แบบ vectorized numpy: พลิก point cloud ให้พื้นอยู่บน แล้วปล่อยผ้าตกทับด้วย
Verlet integration + spring constraint

มีสองปัญหาเฉพาะข้อมูลสแกนจุดเดียวที่ต้องจัดการโดยตรง —
ไกลจากเครื่องสแกน พื้นถูกบังจนจุดต่ำสุดในเซลล์กลายเป็นกิ่งไม้สูงหลายเมตร
(แก้ด้วยการเทียบกับ median ในรัศมี 5 m) และเลยรัศมีที่สแกนถึง ไม่มีจุดพื้นเลย
จึงต้องทำ **valid mask** ทำเครื่องหมายว่าใช้ไม่ได้ แทนที่จะเดาค่าขึ้นมา

**3 · หาลำต้นด้วย circle stacking** — ตัดเป็นชั้นหนา 20 cm ตั้งแต่ 0.6 ถึง 2.6 m
แต่ละชั้นทำ DBSCAN 2D แล้ว RANSAC fit วงกลม จากนั้นเชื่อมวงกลมที่ซ้อนกันในแนวตั้ง
ด้วย union-find ลำต้นหนึ่งต้องปรากฏอย่างน้อย 4 ชั้น

การใช้ slice หนาชั้นเดียวแล้ว project ลง 2D (วิธีที่ดูตรงไปตรงมาที่สุด) ใช้ไม่ได้
เพราะลำต้นที่เอียงหรือเรียวจะกลายเป็นวงแหวนหนาจนหาวงกลมไม่ได้ และลำต้นข้างเคียงจะรวมกัน
ส่วนใบไม้สุ่ม ๆ ไม่มีทางสร้างวงกลมที่ซ้อนตรงกันได้ 4 ชั้น — นี่คือเหตุผลที่วิธีนี้คัดกรองได้ดี

**4 · แยกต้นด้วย geodesic growing** — สร้าง kNN graph บน voxel 15 cm แล้วทำ
**multi-source Dijkstra** จากทุกจุดของ seed โดยให้การเดินทางแนวตั้งถูกกว่าแนวราบ
เส้นทางจึงวิ่งขึ้นลำต้นก่อนแตกออกเรือนยอด

การใช้ระยะ geodesic แทน Euclidean คือสิ่งที่ทำให้แยกเรือนยอดที่ซ้อนกันได้ถูกต้อง —
จุดบนกิ่งอยู่ "ใกล้ผ่านเนื้อไม้" กับลำต้นของตัวเอง แม้เรือนยอดของต้นข้าง ๆ
จะอยู่ใกล้กว่าในเชิงระยะทางตรง

**5 · การวัด** — ความสูงใช้ percentile 99.5 ไม่ใช่ค่าสูงสุด เพื่อไม่ให้ noise จุดเดียวทำให้เพี้ยน
DBH fit บน slice ที่ **ตั้งฉากกับแกนลำต้น** ไม่ใช่ตั้งฉากกับแนวดิ่ง —
ต้นที่เอียง 30° ถ้าตัดตามแนวราบจะได้ค่าใหญ่เกินจริง 15%
ใช้ RANSAC แล้วตามด้วย geometric least squares (soft-L1) เพราะ algebraic fit
จะ bias เมื่อเห็นลำต้นแค่ด้านเดียว ซึ่งเป็นเรื่องปกติของการสแกนจุดเดียว

**6 · Features และการจัดกลุ่ม** — 51 features ต่อต้น (ขนาด, สัดส่วนเรือนยอด,
percentile ความสูงสัมพัทธ์, โปรไฟล์ความหนาแน่นและรัศมี, PCA shape ของทั้งต้นและเฉพาะเรือนยอด)
พร้อม K-means clustering เป็นตัวแทนชั่วคราวจนกว่าจะมี label ชนิดต้นไม้จริง

**7 · รูป QC** — ทุกขั้นตอนมีรูปกำกับ **ห้ามเชื่อตัวเลขโดยไม่ดูรูป**
บั๊กทุกตัวที่เจอตอนสร้าง pipeline นี้โผล่ในรูปทั้งหมด และไม่มีตัวไหนโผล่ในตัวเลขเลย

---

## เกณฑ์คุณภาพ

| ระดับ | เงื่อนไข |
|---|---|
| **good** | axis fit + `arc ≥ 0.60` + residual ≤ 2 cm + ≥ 6 ชั้น + สองวิธีตรงกัน (±60%) |
| **fair** | axis fit + `arc ≥ 0.35` + residual ≤ 3 cm + สองวิธีตรงกัน (±120%) |
| **poor** | นอกเหนือจากนั้น |

`dbh_arc` คือตัวชี้วัดที่สำคัญที่สุด ถ้าเห็นลำต้นน้อยกว่า 1/3 ของเส้นรอบวง
วงกลมจะไถลออกไปตามส่วนโค้งได้อย่างอิสระ — ต้นขนาด 14 cm จึงถูกรายงานเป็น 45 cm ได้
และ **residual ยังคงต่ำตลอด** จึงจับด้วย residual อย่างเดียวไม่ได้

---

## เทียบกับข้อมูลภาคสนาม

```bash
python evaluate_against_reference.py field_plot.csv --quality good --max-dist 1.5
```

ใช้ได้ทั้งกับข้อมูลสำรวจภาคสนามและกับผลจาก tool อื่น (FSCT, TreeLS, TreeLearn)
จับคู่ด้วย optimal assignment (Hungarian) ไม่ใช่ greedy nearest —
ในแปลงที่ลำต้นห่างกันราว 2 m วิธี greedy จะจับคู่ผิดและทำให้ค่า error พองขึ้น

รายงาน precision / recall / F1 / commission / omission และ bias, RMSE, MAE, R²
ของ DBH กับความสูง ไฟล์ reference ต้องมีคอลัมน์ `x`, `y` ในพิกัดเดียวกับ point cloud
ถ้าชื่อคอลัมน์ไม่ตรงใช้ `--map DBH=dbh_cm --map X=x` หรือจับคู่ในหน้า Validation ของเว็บ

---

## หน้าเว็บ

```bash
python serve.py --port 8000 --host 127.0.0.1
```

Backend เป็น FastAPI, frontend เป็น HTML/CSS/JS ธรรมดา ไม่มี build step
`plotly.min.js` คัดลอกมาจาก package ที่ติดตั้งไว้ตอน start ครั้งแรก
หน้าเว็บจึงทำงาน offline ได้และไม่เรียก CDN เลย

เจ็ดหน้า: **Overview** (สถิติแปลงและดาวน์โหลด) · **Stem map** (แผนผังลำต้น hover ดูค่าได้) ·
**3D view** (หมุนดู point cloud ระบายสีตามต้น/ความสูง/คุณภาพ) ·
**Trees** (ตารางเรียงและกรองได้ คลิกแถวเพื่อดูหน้าตัดที่ fit DBH จริง) ·
**Species** · **Validation** · **QC figures**

หน้า 3D แสดงจุดสุ่มตัวอย่าง (ค่าตั้งต้น 150,000 จาก 6.5 ล้านจุด)
พอสำหรับตรวจว่าการแยกต้นสมเหตุสมผลไหม ถ้าต้องการความละเอียดเต็ม
ให้เปิดไฟล์ `.las` ที่ export ออกมาใน CloudCompare แล้วเลือก scalar field `tree_id`

### ใช้งานให้เหมือนแอปบนเครื่อง

เบราว์เซอร์เปิดโปรเซสบนเครื่องไม่ได้ แอปที่ติดตั้งแล้วจึงไม่มีทางสตาร์ตเซิร์ฟเวอร์ของตัวเองได้
`launch.py` คือส่วนที่ขาดไป — มันเปิดเซิร์ฟเวอร์ให้ถ้ายังไม่ได้เปิด รอจนพร้อม แล้วเปิดหน้าเว็บให้

```bash
python launch.py                    # เปิด (ถ้ายังไม่เปิด) แล้วเปิดหน้าเว็บ
python launch.py --stop             # ปิดเซิร์ฟเวอร์ที่รันอยู่เบื้องหลัง
python launch.py --status           # รันอยู่ไหม pid อะไร log อยู่ที่ไหน
python launch.py --port 8080        # ถ้าโปรแกรมอื่นใช้ port 8000 อยู่
python launch.py --install-startup  # ให้เปิดเองอัตโนมัติตอน login
```

กดซ้ำได้ไม่มีปัญหา — ถ้าเซิร์ฟเวอร์เปิดอยู่แล้วมันจะข้ามไปเปิดหน้าเว็บเลย
เซิร์ฟเวอร์ถูกสตาร์ตแบบ detached และไม่มีหน้าต่าง จึงไม่มีหน้าต่างดำค้างไว้
และ output ทั้งหมดไปลง `.cache/server.log` — ถ้าไม่มีไฟล์นี้ เวลาเซิร์ฟเวอร์พังตอนสตาร์ตจะไม่เห็นอะไรเลย

บน Windows:

```
forest_ai.bat                       ดับเบิลคลิกเพื่อเปิด
stop.bat                            ดับเบิลคลิกเพื่อปิด
python tools/make_shortcuts.py      สร้างช็อตคัต Desktop + Start Menu พร้อมไอคอน
```

หน้าเว็บยังเป็น PWA ด้วย เบราว์เซอร์จึงติดตั้งเป็นแอปได้ — เปิดแอปแล้วกด
**Install as an app** ในแถบซ้าย จะได้หน้าต่างของตัวเอง มีไอคอน ไม่มีแถบที่อยู่
ถ้าใช้คู่กับ `--install-startup` ก็เหลือแค่กดไอคอนอย่างเดียวตลอดไป

สองเรื่องที่ต้องรู้:

- **ยังต้องมีเซิร์ฟเวอร์อยู่ดี** การคำนวณทุกอย่างอยู่ในโปรเซส Python
  service worker แคชแค่ตัวหน้าเว็บ ไม่ได้แคชการคำนวณ
  ถ้าเปิดแอปตอนเซิร์ฟเวอร์ไม่ได้รัน จะขึ้นหน้าบอกว่าต้องรันคำสั่งอะไร แทนที่จะขึ้น error ของเบราว์เซอร์
- **ติดตั้งได้เฉพาะผ่าน `localhost`** เบราว์เซอร์บังคับว่า service worker ต้องอยู่ใน secure context
  ซึ่ง `http://` ไปยัง IP ในวง LAN ไม่นับ เครื่องอื่นจึงเปิดใช้ได้แต่ติดตั้งเป็นแอปไม่ได้

ตัวหน้าเว็บขนาด ~5 MB (ส่วนใหญ่คือ `plotly.min.js`) ถูก precache ไว้ เปิดครั้งต่อไปจึงขึ้นทันที
ส่วนทุกอย่างใต้ `/api/` ไม่เคยถูกแคชเลย เพราะเป็นข้อมูลเฉพาะ session และเปลี่ยนทุกครั้งที่รัน

### REST API

หน้าเว็บคุยกับ backend ผ่าน REST ล้วน ๆ เรียกจากสคริปต์หรือ QGIS ได้เหมือนกัน
เอกสารอัตโนมัติที่ `/api/docs`

```
GET  /api/clouds  /api/params  /api/config  /api/header?las=
POST /api/upload                        stream และจำกัดขนาด
POST /api/run     GET /api/job          เริ่มงาน / ดู progress
GET  /api/summary /api/trees /api/species
GET  /api/figure/{cloud,stemmap,inventory,dbh_slice/{id},tree3d/{id}}
POST /api/evaluate  /api/write_outputs
GET  /api/download/{trees,features}.csv
```

ทุก request แนบ header `X-Session-Id` — ผลลัพธ์ ไฟล์ที่อัปโหลด และ output
แยกตาม session ทั้งหมด

### การตั้งค่า

| ตัวแปร | ค่าตั้งต้น | ความหมาย |
|---|---|---|
| `FAI_MAX_SESSIONS` | 2 | จำนวนผลลัพธ์ที่เก็บใน memory พร้อมกัน (~700 MB ต่อชุด) |
| `FAI_MAX_UPLOAD_MB` | 300 | ขนาดไฟล์อัปโหลดสูงสุด |
| `FAI_ALLOW_LOCAL_CLOUDS` | true | ให้เลือกไฟล์ `.las` ที่อยู่ข้างเซิร์ฟเวอร์ได้ |
| `FAI_ALLOW_SEGMENTED_LAS` | true | อนุญาตให้เขียนไฟล์ segmented ~250 MB |

Docker image ตั้งสองตัวท้ายเป็น `false` เพื่อไม่ให้เซิร์ฟเวอร์สาธารณะ
เปิดเผยไฟล์ในเครื่องและไม่ให้ดิสก์เต็ม

**รันได้ทีละ worker เท่านั้น** — ผลลัพธ์เป็น numpy ~700 MB ที่อยู่ใน memory ของ process
แชร์ข้าม worker ไม่ได้ ทั้ง `serve.py` และ Dockerfile บังคับไว้แล้ว
การรองรับผู้ใช้หลายคนมาจากตาราง session ส่วนงานหนักรันบน thread pool ช่องเดียว
ผู้ใช้คนที่สองจึงเข้าคิวรอ ไม่ใช่แย่ง CPU กัน

---

## การ deploy

```bash
docker build -t forest_ai .
docker run -p 7860:7860 forest_ai
```

โฮสต์ไหนที่รัน container ได้ก็ใช้ได้ทั้งหมด ส่วน `app_port: 7860`
และบล็อก YAML ด้านบนของ `README.md` มีไว้สำหรับ **Hugging Face Spaces**

```bash
git remote add space https://huggingface.co/spaces/<user>/<space>
git push space main
```

> **Hugging Face ไม่ฟรีสำหรับงานนี้แล้ว** ตั้งแต่กรกฎาคม 2026 การโฮสต์ Gradio
> หรือ Docker Space บน `cpu-basic` แบบฟรีต้องสมัคร PRO ($9/เดือน)
> เหลือแค่ Static Space ที่ยังฟรี ซึ่งรัน Python ไม่ได้
> วางแผนเผื่อไว้ หรือเลือกโฮสต์อื่น — `Dockerfile` ตัวเดียวกันใช้ได้ทุกที่

การประเมินสเปกสำหรับโฮสต์ใด ๆ: pipeline ใช้หน่วยความจำสูงสุดราว **700 MB RSS**
บน point cloud 15.8 ล้านจุด ดังนั้นเครื่องที่มี RAM 512 MB **ไม่พอ**
ส่วน CPU 2 cores จะทำให้เวลารัน ~90 วินาที (วัดบน 8 cores) กลายเป็นไม่กี่นาที

---

## ข้อจำกัด

1. **การถูกบัง (occlusion) คือเพดาน ไม่ใช่ algorithm** — บนชุดข้อมูลที่ใช้พัฒนา
   คุณภาพการวัดสัมพันธ์กับระยะห่างจากเครื่องสแกนแทบจะสมบูรณ์แบบ
   (มัธยฐาน 17 m สำหรับ *good*, 24 m สำหรับ *fair*, 31 m สำหรับ *poor*)
   และเกิน ~28 m แทบไม่มีต้นไหนได้ `arc > 0.5` เลย
   ต้นไม้ที่เห็นชัดใน point cloud เลยระยะนั้นไปยังตรวจไม่พบ
   **วิธีแก้เดียวคือสแกนเพิ่มหลายจุดแล้ว register เข้าด้วยกัน**
   ไม่มี algorithm ไหนกู้คืนจุดที่ไม่เคยถูกยิงกลับมาได้

2. **ค่าต่อเฮกตาร์ต้องหารด้วยพื้นที่ที่มีต้นไม้จริง** — โปรแกรมใช้ convex hull
   ของลำต้นที่ตรวจพบ ไม่ใช่ขอบเขตของ DTM ที่ใช้ได้
   เพราะบนข้อมูลชุดนี้ DTM ครอบคลุมพื้นที่โล่งด้วย ถ้าใช้จะทำให้ค่าความหนาแน่น
   และ basal area ต่ำกว่าความจริงถึงสามเท่า

3. **ยังจำแนกชนิดต้นไม้ไม่ได้ เตรียมไว้เฉย ๆ** — feature extractor และโครง
   RandomForest แบบ cross-validated พร้อมแล้ว แต่ยังไม่มี label
   `structural_group` เป็นการจัดกลุ่มตามรูปทรงด้วย K-means
   สองชนิดที่มีโครงสร้างเหมือนกันจะอยู่กลุ่มเดียวกัน และชนิดเดียวกันจะแตกเป็นหลายกลุ่ม
   ถ้าบางต้นถูกกดการเจริญเติบโต — ประโยชน์ของมันคือช่วยลดงานสำรวจ
   ระบุชนิดแค่ 10–15 ต้นต่อกลุ่มแทนที่จะทั้งหมด

4. **`dbh_rmse_cm` แยกแยะได้ไม่ดี** — กระจุกอยู่ที่ 0.9–1.2 cm แทบทุกต้น
   เพราะ soft-L1 loss saturate ที่ `f_scale=0.02` ให้ใช้ `dbh_arc` กรองแทน

5. **ยังไม่ได้วัดความแม่นยำเทียบข้อมูลจริง** — เกณฑ์ (`arc ≥ 0.60 / 0.35`)
   ตั้งจากหลักการ ไม่ได้มาจากการ calibrate
   จนกว่าจะเทียบกับต้นไม้ที่วัดในสนาม 30–50 ต้น ค่า RMSE ที่แท้จริงยังไม่ทราบ

---

## โครงสร้างโค้ด

```
forest_ai/          ตัว pipeline — ไม่มี web framework อยู่ในนี้เลย
  config.py         พารามิเตอร์ทั้งหมดใน dataclass เดียว
  pipeline.py       ทุกขั้นรวมเป็นฟังก์ชันเดียว พร้อม progress callback
  las_io.py         อ่าน/เขียน .las แบบ chunk ประหยัด memory
  preprocess.py     ลบ noise และ downsample ด้วย voxel key
  ground.py         CSF, DTM พร้อม validity mask, height normalization
  fitting.py        RANSAC circle, geometric refine, arc coverage, PCA axis
  segment.py        circle stacking, union-find, geodesic segmentation
  measure.py        DBH/ความสูง/เรือนยอดต่อต้น และธงคุณภาพ
  features.py       51 features, K-means, โครง RandomForest
  evaluate.py       optimal-assignment matching และสถิติ error
  qc.py             รูป QC 5 รูป (matplotlib)
  webviz.py         กราฟ interactive (plotly)
web/                ชั้น HTTP บาง ๆ ครอบ pipeline.py
  server.py  sessions.py  params.py  vendor.py
  static/           index.html, app.css, app.js, sw.js, manifest, icons
tools/              make_icons.py, make_shortcuts.py
launch.py           ตัวเปิดแบบสตาร์ตให้ถ้ายังไม่เปิด (forest_ai.bat / stop.bat เรียกตัวนี้)
serve.py  run_pipeline.py  evaluate_against_reference.py  Dockerfile
```

การปรับจูนทำที่ `forest_ai/config.py` (หรือ slider ในหน้าเว็บ)
DTM cache ไว้ใน `.cache/` โดย key ผูกกับ path + ขนาด + mtime ของไฟล์
และพารามิเตอร์ทุกตัวที่มีผลต่อ DTM ชี้ไปไฟล์อื่นจึงไม่มีทางหยิบ DTM ผิดมาใช้

**อยากแก้อะไรที่ไหน**

| อยากแก้ | ไฟล์ |
|---|---|
| ค่าพารามิเตอร์ default | `forest_ai/config.py` |
| เพิ่ม slider ในหน้าเว็บ | `web/params.py` |
| หน้าตา / layout | `web/static/app.css` |
| พฤติกรรมหน้าเว็บ | `web/static/app.js` |
| logic การวิเคราะห์ | `forest_ai/segment.py`, `measure.py`, `ground.py` |

---

## หมายเหตุ

พัฒนาและทดสอบกับข้อมูล TLS จุดสแกนเดียวของสวนป่า 15.8 ล้านจุด
(ขอบเขต 167 × 142 m, ~666 จุด/ตร.ม., ไม่มี RGB และ intensity)
ไฟล์ point cloud ไม่ได้อยู่ใน repository นี้ เพราะใหญ่เกินกว่าที่ GitHub รับได้
