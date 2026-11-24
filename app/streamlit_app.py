"""Ứng dụng Streamlit: chẩn đoán bệnh lá táo / cà chua + xem kết quả thực nghiệm và dữ liệu.

  .venv/bin/streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import ROOT, auto_predict, available_models  # noqa: E402

RESULTS = Path(os.environ.get("PLANTDX_RESULTS", ROOT / "results"))
SPECIES_VI = {"apple": "Táo", "tomato": "Cà chua"}
DOMAIN_VI = {"pv": "PlantVillage", "pp": "Plant Pathology", "pd": "PlantDoc (lá cắt)", "pdfull": "PlantDoc (ảnh nguyên)"}
CONFIG_VI = {"convnext_tiny": "ConvNeXt-T (teacher)", "resnet50": "ResNet-50", "mnv3_erm": "MNv3-L ERM",
             "mnv3_mixstyle": "MNv3-L + MixStyle", "mnv3_kd": "MNv3-L + KD", "mnv3_cons": "MNv3-L + nhất quán nền",
             "proposed": "Đề xuất (KD + nhất quán nền)"}

st.set_page_config(page_title="Chẩn đoán bệnh lá", page_icon="🌿", layout="wide")


@st.cache_resource
def load_models():
    return available_models()


@st.cache_data
def sample_images(per_domain: int = 2) -> dict[str, str]:
    """Vài ảnh test có nhãn (mỗi loài × miền) để thử nhanh, kèm một ảnh không phải táo/cà chua."""
    out = {}
    d = ROOT / "data" / "processed"
    names = {"plantvillage": "PlantVillage", "plantpathology": "Plant Pathology", "plantdoc_crop": "PlantDoc"}
    for sp in ["apple", "tomato"]:
        f = d / "species" / f"manifest_{sp}.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        m = m[(m.split == "test") & m.dataset.isin(list(names))]
        for r in m.groupby("dataset").sample(per_domain, random_state=1).itertuples():
            out[f"{SPECIES_VI[sp]} · {names[r.dataset]} · {r.unified_class.split('_', 1)[1]}"] = str(d.parent / r.path)
    base = d / "manifest.csv"
    if base.exists():
        m = pd.read_csv(base)
        m = m[(m.split == "test") & (m.dataset == "plantvillage") & m.unified_class.str.startswith(("grape", "corn"))]
        for r in m.sample(2, random_state=1).itertuples():
            out[f"Không phải táo/cà chua · {r.unified_class}"] = str(d.parent / r.path)
    return out


def tab_diagnose():
    models = load_models()
    if not models:
        st.warning("Chưa có mô hình triển khai trong `models/`. Huấn luyện rồi chạy:\n\n"
                   "`PYTHONPATH=src .venv/bin/python -m plantdx.deploy --ckpt runs/<loài>/<thiết lập>/proposed/seed0/best.pt`")
        return
    opts = ["Tự động nhận loài"] + [SPECIES_VI[s] for s in models]
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Loài cây", opts, horizontal=False)
        samples = sample_images()
        srcs = ["Tải ảnh lên", "Chụp bằng camera"] + (["Ảnh mẫu (tập test)"] if samples else [])
        src = st.radio("Nguồn ảnh", srcs, horizontal=True)
        if src == "Tải ảnh lên":
            file = st.file_uploader("Ảnh một chiếc lá", type=["jpg", "jpeg", "png", "webp"])
        elif src == "Chụp bằng camera":
            file = st.camera_input("Chụp một chiếc lá, đặt lá ở giữa khung")
        else:
            key = st.selectbox("Ảnh mẫu (nhãn thật)", list(samples))
            file = samples[key]
        st.caption("Mẹo: chụp một lá, đủ sáng, lá chiếm phần lớn khung hình, mặt có vết bệnh hướng lên.")
    if file is None:
        with c2:
            st.info("Chọn hoặc chụp một ảnh lá để bắt đầu.")
        return
    img = Image.open(file)
    if mode == "Tự động nhận loài":
        pred, all_preds = auto_predict(models, img)
        if all(p.status == "not_leaf" for p in all_preds.values()):
            pred = max(all_preds.values(), key=lambda p: p.energy - models[p.species].meta["tau_energy"])
    else:
        sp = {v: k for k, v in SPECIES_VI.items()}[mode]
        pred, all_preds = models[sp].predict(img), None
    meta = models[pred.species].meta
    with c2:
        a, b = st.columns([1, 1])
        a.image(img, width="stretch")
        with b:
            if pred.status == "not_leaf":
                st.error(f"**Có vẻ không phải lá {'táo hoặc cà chua' if all_preds else SPECIES_VI[pred.species].lower()}.**\n\n"
                         "Hãy chụp lại một chiếc lá rõ nét, chiếm phần lớn khung hình.")
            elif pred.status == "uncertain":
                st.warning(f"**Chưa chắc chắn** – có thể là: **{pred.name_vi}** ({pred.prob:.0%}).\n\n"
                           "Hãy chụp lại gần hơn, đủ sáng, hoặc chụp mặt lá có vết bệnh.")
            else:
                st.success(f"### {pred.name_vi}\nĐộ tin cậy (đã hiệu chỉnh): **{pred.prob:.0%}**")
            if all_preds:
                st.caption("Loài nhận được: **" + SPECIES_VI[pred.species] + "**")
            top = pd.DataFrame({"Bệnh": meta["names_vi"], "Xác suất": pred.probs}).sort_values("Xác suất", ascending=False)
            st.bar_chart(top.head(3).set_index("Bệnh"), horizontal=True, height=170)
            st.caption(f"Xử lý {pred.ms:.0f} ms trên CPU · mô hình {meta['model']} INT8 · {meta['size_mb']['int8']:.1f} MB")
        with st.expander("Chi tiết kỹ thuật"):
            rows = [{"Loài": SPECIES_VI[p.species], "Năng lượng": round(p.energy, 2),
                     "Ngưỡng τ_E": round(models[p.species].meta["tau_energy"], 2), "Xác suất lớn nhất": round(p.prob, 3),
                     "Ngưỡng τ_c": models[p.species].meta["tau_conf"], "Kết luận": p.status}
                    for p in (all_preds.values() if all_preds else [pred])]
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.write(f"Nhiệt độ hiệu chỉnh T* = {meta['T']:.2f}. Năng lượng = T·logsumexp(logits/T); "
                     "ảnh có năng lượng dưới τ_E bị coi là ảnh lạ. Ảnh có xác suất dưới τ_c được báo \"chưa chắc\".")
    st.caption("Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của cán bộ bảo vệ thực vật.")


def tab_results():
    f = RESULTS / "summary.csv"
    if not f.exists():
        st.info("Chưa có kết quả. Chạy `bash scripts/run_experiments.sh` (hoặc `PYTHONPATH=src .venv/bin/python -m plantdx.summarize` nếu đã có runs/).")
        return
    s = pd.read_csv(f)
    c1, c2 = st.columns(2)
    sp = c1.selectbox("Loài", sorted(s.species.unique()), format_func=lambda x: SPECIES_VI.get(x, x))
    sub = s[s.species == sp]
    setting = c2.selectbox("Thiết lập", sorted(sub.setting.unique()))
    sub = sub[sub.setting == setting].copy()
    sub["Cấu hình"] = sub.config.map(lambda c: CONFIG_VI.get(c, c))
    sub["Miền"] = sub.domain.map(DOMAIN_VI) + sub.heldout.map({True: " (chưa thấy)", False: ""})
    piv = sub.pivot_table(index="Cấu hình", columns="Miền", values="f1_mean") * 100
    st.subheader("F1 macro (%) trên tập test, trung bình qua các seed")
    st.dataframe(piv.style.format("{:.1f}").highlight_max(axis=0, color="#c8e6c9"), width="stretch")
    st.bar_chart(piv, stack=False, height=320)
    extra = [c for c in ["ece_after_mean", "aurc_mean", "coverage_mean", "acc_answered_mean", "ood_auroc_mean"] if c in sub]
    if extra:
        st.subheader("Độ tin cậy và lọc ảnh lạ")
        st.dataframe(sub[["Cấu hình", "Miền"] + extra].round(3), hide_index=True, width="stretch")
    md = RESULTS / "summary.md"
    if md.exists():
        st.download_button("Tải bảng Markdown", md.read_text(), file_name="summary.md")


def tab_data():
    sp = st.radio("Loài", ["apple", "tomato"], format_func=SPECIES_VI.get, horizontal=True)
    d = ROOT / "data" / "processed" / "species"
    f = d / f"stats_{sp}.csv"
    if not f.exists():
        st.info("Chưa chuẩn bị dữ liệu. Chạy `scripts/prepare_species.py`.")
        return
    st.subheader("Số ảnh theo lớp × miền × tập")
    st.dataframe(pd.read_csv(f, header=[0, 1], index_col=0), width="stretch")
    img = d / "qa" / f"samples_{sp}.jpg"
    if img.exists():
        st.subheader("Ảnh mẫu sau làm sạch")
        st.image(str(img), width="stretch")
    st.caption("Chi tiết quy trình làm sạch: tài liệu LamSachDuLieu_TaoCaChua.docx (thư mục BaoCao).")


def tab_about():
    st.markdown("""
**Mô hình đề xuất**: học sinh **MobileNetV3-Large** (5,5 triệu tham số, chạy trên điện thoại) được huấn luyện với

- **Chưng cất tri thức** từ thầy **ConvNeXt-Tiny** (28 triệu tham số) đã fine-tune trên cùng các miền huấn luyện;
- **Loss nhất quán nền**: cùng một lá PlantVillage đặt trên nền thực địa tổng hợp phải cho cùng dự đoán;
- **Lấy mẫu cân bằng** theo miền (PlantVillage / Plant Pathology / PlantDoc) và lớp.

`loss = (1−α)·CE + α·T²·KL(thầy ‖ học sinh) + β·KL(p(x) ‖ p(x_nền))`, với α = 0,5, T = 4, β = 1.

**Triển khai tin cậy**: hiệu chỉnh nhiệt độ T*, ngưỡng năng lượng τ_E để từ chối ảnh không phải lá,
ngưỡng tin cậy τ_c để báo "chưa chắc". Mô hình xuất ONNX INT8 (~5 MB) dùng chung cho app Android và web này.
""")
    for sp, m in load_models().items():
        meta = m.meta
        st.write(f"**{SPECIES_VI[sp]}** – cấu hình `{meta['config']}`, thiết lập `{meta['setting']}`, "
                 f"{len(meta['classes'])} lớp, T* = {meta['T']:.2f}, τ_E = {meta['tau_energy']:.2f}, τ_c = {meta['tau_conf']:.2f}, "
                 f"{meta['size_mb']['int8']:.1f} MB, {meta['latency_ms_1thread']['mean_ms']:.1f} ms/ảnh (1 luồng).")


st.title("🌿 Chẩn đoán bệnh lá táo và cà chua")
t1, t2, t3, t4 = st.tabs(["Chẩn đoán", "Kết quả thực nghiệm", "Dữ liệu", "Mô hình"])
with t1:
    tab_diagnose()
with t2:
    tab_results()
with t3:
    tab_data()
with t4:
    tab_about()
