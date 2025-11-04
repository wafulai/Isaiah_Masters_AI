# smart_clustering_pro.py
# =============================================================
# Smart Auto-Clustering Advisor — Pro Edition (extended)
# Adds: Comparison score heatmap + PDF report generation
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import io
import os
from datetime import datetime

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import utils

# Optional: HDBSCAN support
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# Page config
st.set_page_config(page_title="Smart Clustering Pro", layout="wide")
st.title("🤖 Smart Auto-Clustering Advisor — Pro")
st.markdown(
    "Side-by-side comparison (K-Means, DBSCAN, Hierarchical) with live Silhouette, Elbow, 2D & 3D PCA (Plotly). Now includes a comparison heatmap and PDF report export."
)
# --- Authors Section ---
st.markdown("""
---
👨‍💻 **Developed by:**
1. Martin  Nyamu
2. Isaiah  Wafula
3. Jackson  Mutava
4. Jeff  
5. Esther  
6. Aaron Muuo
7. Johnrich  
8. Erick 

📘 *OUK Masters in AI GROUP 4 PROJECT: Smart Clustering Pro – Interactive Clustering Explorer (2025)*
---
""")

# ---------------------------
# Data loading & preview
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload CSV (or leave empty to use Iris)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded")
else:
    st.info("Using built-in Iris dataset")
    df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

st.subheader("Dataset preview")
st.dataframe(df.head())

# ---------------------------
# Feature selection
# ---------------------------
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns to run clustering.")
    st.stop()

st.sidebar.header("⚙️ Features & Global Settings")
features = st.sidebar.multiselect("Select numeric features to cluster on", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))])
if len(features) < 2:
    st.warning("Choose at least two numeric features.")
    st.stop()

X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# K-Means controls (live)
# ---------------------------
st.sidebar.subheader("🔹 K-Means (Live Explorer)")
k_live = st.sidebar.slider("K for live K-Means", min_value=2, max_value=12, value=3, step=1)
auto_k_opt = st.sidebar.checkbox("Auto-find best k (2–12) by silhouette", value=False)

if auto_k_opt:
    K_range_auto = range(2, 13)
    sils = []
    for k_try in K_range_auto:
        try:
            sils.append(silhouette_score(X_scaled, KMeans(n_clusters=k_try, random_state=42).fit_predict(X_scaled)))
        except Exception:
            sils.append(-1)
    best_k_auto = K_range_auto[int(np.argmax(sils))]
    st.sidebar.success(f"Auto best k = {best_k_auto} (sil={max(sils):.3f})")
    k_live = best_k_auto

# Elbow interactive range
st.sidebar.subheader("Elbow Plot Range")
elbow_max_k = st.sidebar.slider("Max K to compute for Elbow", min_value=5, max_value=20, value=10, step=1)

# ---------------------------
# DBSCAN controls
# ---------------------------
st.sidebar.subheader("🔸 DBSCAN")
dbscan_auto = st.sidebar.checkbox("Auto-find DBSCAN eps (scan)", value=False)
dbscan_min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 30, 5)

if dbscan_auto:
    eps_grid = np.round(np.linspace(0.05, 2.0, 40), 2)
    best_eps = 0.5
    best_sil = -2
    for e in eps_grid:
        labels_try = DBSCAN(eps=e, min_samples=dbscan_min_samples).fit_predict(X_scaled)
        if len(set(labels_try)) > 1:
            try:
                s = silhouette_score(X_scaled, labels_try)
                if s > best_sil:
                    best_sil = s
                    best_eps = e
            except Exception:
                pass
    st.sidebar.success(f"Auto eps ≈ {best_eps} (sil={best_sil:.3f})")
    dbscan_eps = best_eps
else:
    dbscan_eps = st.sidebar.slider("DBSCAN eps", 0.05, 2.5, 0.5, step=0.01)

# ---------------------------
# Hierarchical controls
# ---------------------------
st.sidebar.subheader("🧩 Hierarchical (Agglomerative)")
hier_clusters = st.sidebar.slider("Hierarchical # clusters", 2, 12, 3)
linkage_method = st.sidebar.selectbox("Linkage method", options=["ward", "complete", "average", "single"])

# ---------------------------
# Optional HDBSCAN
# ---------------------------
if HDBSCAN_AVAILABLE:
    st.sidebar.subheader("🟢 HDBSCAN (optional)")
    hdb_min_cluster_size = st.sidebar.slider("HDBSCAN min_cluster_size", 2, 50, 5)

# ---------------------------
# Compute clusterings
# ---------------------------
# K-Means live
kmeans_model = KMeans(n_clusters=k_live, random_state=42)
k_labels = kmeans_model.fit_predict(X_scaled)
df["KMeans_Cluster"] = k_labels

# DBSCAN
dbscan_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
db_labels = dbscan_model.fit_predict(X_scaled)
df["DBSCAN_Cluster"] = db_labels

# Hierarchical
hier_model = AgglomerativeClustering(n_clusters=hier_clusters, linkage=linkage_method)
hier_labels = hier_model.fit_predict(X_scaled)
df["Hier_Cluster"] = hier_labels

# HDBSCAN if available
if HDBSCAN_AVAILABLE:
    hdb_model = hdbscan.HDBSCAN(min_cluster_size=hdb_min_cluster_size)
    hdb_labels = hdb_model.fit_predict(X_scaled)
    df["HDBSCAN_Cluster"] = hdb_labels

# ---------------------------
# Metrics helpers (safe)
# ---------------------------
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def safe_metrics(Xs, labels):
    unique = set(labels)
    if len(unique) <= 1 or (len(unique) == 2 and -1 in unique):
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan, "Noise%": np.nan}
    noise = list(labels).count(-1) / len(labels) * 100
    return {
        "Silhouette": float(np.round(silhouette_score(Xs, labels), 4)),
        "Davies-Bouldin": float(np.round(davies_bouldin_score(Xs, labels), 4)),
        "Calinski-Harabasz": float(np.round(calinski_harabasz_score(Xs, labels), 4)),
        "Noise%": float(np.round(noise, 2))
    }

metrics = {
    "K-Means": safe_metrics(X_scaled, k_labels),
    "DBSCAN": safe_metrics(X_scaled, db_labels),
    "Hierarchical": safe_metrics(X_scaled, hier_labels),
}
if HDBSCAN_AVAILABLE:
    metrics["HDBSCAN"] = safe_metrics(X_scaled, hdb_labels)

metrics_df = pd.DataFrame(metrics).T

# ---------------------------
# Top summary & Download
# ---------------------------
st.subheader("Quick Metrics Summary")
st.dataframe(metrics_df.style.highlight_max(subset=["Silhouette", "Calinski-Harabasz"], color="lightgreen")
                         .highlight_min(subset=["Davies-Bouldin"], color="#ffcccc"))

# Download results
st.download_button("💾 Download results (CSV)", df.to_csv(index=False).encode("utf-8"), "clustered_results.csv", "text/csv")

# ---------------------------
# Elbow plot (interactive range)
# ---------------------------
st.subheader("Elbow Method (Inertia vs K)")
Klist = list(range(1, elbow_max_k + 1))
inertias = []
for k_ in Klist:
    try:
        inertias.append(KMeans(n_clusters=k_, random_state=42).fit(X_scaled).inertia_)
    except Exception:
        inertias.append(np.nan)

fig_elb = px.line(x=Klist, y=inertias, markers=True, labels={"x": "K", "y": "Inertia"})
fig_elb.update_layout(title="Elbow Method", xaxis=dict(dtick=1))
st.plotly_chart(fig_elb, use_container_width=True)

# ---------------------------
# Side-by-side comparison panel (KMeans | DBSCAN | Hierarchical)
# ---------------------------
st.header("Side-by-side Comparison")

col_km, col_db, col_hier = st.columns(3)

# Compute PCA once (3 components) for consistent plots
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])

def plot_algo_panel(col, title, labels, pca_df, show_sil=True):
    with col:
        st.subheader(title)
        pca_df_local = pca_df.copy()
        pca_df_local["cluster"] = labels.astype(str)
        # 2D Plotly
        fig2d = px.scatter(pca_df_local, x="PC1", y="PC2", color="cluster", title=f"{title} — 2D PCA",
                           color_discrete_sequence=px.colors.qualitative.Vivid)
        st.plotly_chart(fig2d, use_container_width=True)
        # 3D Plotly
        fig3d = px.scatter_3d(pca_df_local, x="PC1", y="PC2", z="PC3", color="cluster",
                              title=f"{title} — 3D PCA", color_discrete_sequence=px.colors.qualitative.Vivid,
                              height=420)
        st.plotly_chart(fig3d, use_container_width=True)
        # Silhouette (matplotlib) and metric display
        if show_sil:
            try:
                vals = silhouette_samples(X_scaled, labels)
                avg = silhouette_score(X_scaled, labels)
                fig, ax = plt.subplots(figsize=(5, 2.6))
                y_lower = 10
                unique_clusters = sorted([c for c in set(labels) if c != -1])
                # include noise cluster if present
                clusters_to_plot = unique_clusters + ([-1] if -1 in set(labels) else [])
                for i, cl in enumerate(clusters_to_plot):
                    cl_vals = vals[labels == cl]
                    cl_vals.sort()
                    size = cl_vals.shape[0]
                    if size == 0:
                        continue
                    y_upper = y_lower + size
                    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_vals, alpha=0.7)
                    ax.text(-0.05, y_lower + 0.5 * size, str(cl))
                    y_lower = y_upper + 10
                ax.axvline(x=avg, color="red", linestyle="--")
                ax.set_xlabel("Silhouette Coefficient")
                ax.set_yticks([])
                ax.set_title(f"Silhouette (avg={avg:.3f})")
                st.pyplot(fig)
            except Exception:
                st.info("Silhouette not available for this algorithm (maybe only one cluster or invalid labels).")
        # Show metrics table for this algorithm
        m = safe_metrics(X_scaled, labels)
        st.markdown("**Metrics**")
        st.write(pd.DataFrame([m], index=[title]))
        st.markdown("---")

# Plot KMeans panel
plot_algo_panel(col_km, f"K-Means (k={k_live})", k_labels, pca_df, show_sil=True)

# Plot DBSCAN panel
plot_algo_panel(col_db, f"DBSCAN (eps={dbscan_eps}, min_samp={dbscan_min_samples})", db_labels, pca_df, show_sil=True)

# Plot Hierarchical panel
plot_algo_panel(col_hier, f"Hierarchical (link={linkage_method}, k={hier_clusters})", hier_labels, pca_df, show_sil=True)

# ---------------------------
# Dendrogram & Hierarchical details
# ---------------------------
st.subheader("Hierarchical Dendrogram (Ward linkage default for dendrogram plotting)")
try:
    Z = linkage(X_scaled, method=linkage_method)
    fig_d, axd = plt.subplots(figsize=(10, 4))
    dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
    axd.set_title(f"Dendrogram ({linkage_method})")
    st.pyplot(fig_d)
except Exception as e:
    st.warning(f"Could not compute dendrogram: {e}")

# ---------------------------
# AI Advisor: combine normalized metrics to pick best algorithm
# ---------------------------
st.subheader("🧠 AI Advisor Recommendation")
def ai_recommend(metrics_df):
    dfc = metrics_df.copy().dropna()
    if dfc.empty:
        return "No valid algorithm metrics to recommend."
    # Normalise each metric (higher is better for silhouette & CH; lower better for DB)
    dfc_norm = dfc.copy()
    dfc_norm["Sil_N"] = dfc_norm["Silhouette"] / dfc_norm["Silhouette"].max()
    dfc_norm["CH_N"] = dfc_norm["Calinski-Harabasz"] / dfc_norm["Calinski-Harabasz"].max()
    dfc_norm["DB_N"] = 1 - (dfc_norm["Davies-Bouldin"] / dfc_norm["Davies-Bouldin"].max())
    dfc_norm["Score"] = dfc_norm[["Sil_N", "CH_N", "DB_N"]].mean(axis=1)
    best = dfc_norm["Score"].idxmax()
    score = dfc_norm.loc[best, "Score"]
    details = dfc.loc[best].to_dict()
    return f"✅ Recommend **{best}** (combined score={score:.3f}) — Details: Silhouette={details['Silhouette']}, DB={details['Davies-Bouldin']}, CH={details['Calinski-Harabasz']}"

st.success(ai_recommend(pd.DataFrame(metrics).T))

# ---------------------------
# NEW: Comparison Score Heatmap (who wins each metric)
# ---------------------------
st.subheader("🔥 Comparison Score Heatmap — Metric Winners")

# Build winners table:
# For Silhouette and Calinski-Harabasz: higher is better (max wins)
# For Davies-Bouldin and Noise%: lower is better (min wins)
metrics_for_heat = metrics_df.copy()
# make sure columns exist
for colm in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Noise%"]:
    if colm not in metrics_for_heat.columns:
        metrics_for_heat[colm] = np.nan

winners = pd.DataFrame(0, index=metrics_for_heat.index, columns=["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Noise%"])
# Silhouette winner (max)
if not metrics_for_heat["Silhouette"].dropna().empty:
    sil_winner = metrics_for_heat["Silhouette"].idxmax()
    winners.loc[sil_winner, "Silhouette"] = 1
# Davies-Bouldin winner (min)
if not metrics_for_heat["Davies-Bouldin"].dropna().empty:
    db_winner = metrics_for_heat["Davies-Bouldin"].idxmin()
    winners.loc[db_winner, "Davies-Bouldin"] = 1
# Calinski-Harabasz winner (max)
if not metrics_for_heat["Calinski-Harabasz"].dropna().empty:
    ch_winner = metrics_for_heat["Calinski-Harabasz"].idxmax()
    winners.loc[ch_winner, "Calinski-Harabasz"] = 1
# Noise% winner (min)
if not metrics_for_heat["Noise%"].dropna().empty:
    noise_winner = metrics_for_heat["Noise%"].idxmin()
    winners.loc[noise_winner, "Noise%"] = 1

# show heatmap with plotly
heat_fig = px.imshow(winners.astype(int),
                     labels=dict(x="Metric", y="Algorithm", color="Win (1=yes)"),
                     x=winners.columns.tolist(),
                     y=winners.index.tolist(),
                     color_continuous_scale=[[0, "lightgray"], [1, "green"]],
                     text_auto=True)
heat_fig.update_layout(height=300, title="Which algorithm wins each metric (1 = winner)")
st.plotly_chart(heat_fig, use_container_width=True)

# ---------------------------
# NEW: PDF report generation (small sample)
# ---------------------------
st.subheader("📄 Generate PDF Report (sample)")

# Helper to save matplotlib figure to PNG bytes
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf

# Prepare static figures for the report:
# 1) Elbow (matplotlib version)
fig_elbow_mpl, ax_elb = plt.subplots(figsize=(6, 3))
ax_elb.plot(Klist, inertias, marker='o')
ax_elb.set_xlabel("K")
ax_elb.set_ylabel("Inertia")
ax_elb.set_title("Elbow Method")
ax_elb.grid(True)
elb_png = fig_to_png_bytes(fig_elbow_mpl)
plt.close(fig_elbow_mpl)

# 2) Silhouette plot for recommended algorithm (AI picks)
# Determine recommended algorithm name
metrics_for_pick = pd.DataFrame(metrics).T
# if all NaN, fallback to K-Means
if metrics_for_pick.dropna().empty:
    recommended_algo = "K-Means"
else:
    rec_text = ai_recommend(metrics_for_pick)
    # extract name between ** **
    import re
    m = re.search(r"\*\*(.+?)\*\*", rec_text)
    recommended_algo = m.group(1) if m else "K-Means"

# create silhouette for recommended algorithm (if possible)
sil_png = None
try:
    if recommended_algo == "K-Means":
        labels_for_sil = k_labels
    elif recommended_algo == "DBSCAN":
        labels_for_sil = db_labels
    elif recommended_algo == "Hierarchical":
        labels_for_sil = hier_labels
    elif recommended_algo == "HDBSCAN" and HDBSCAN_AVAILABLE:
        labels_for_sil = hdb_labels
    else:
        labels_for_sil = k_labels

    if len(set(labels_for_sil)) > 1:
        sil_vals = silhouette_samples(X_scaled, labels_for_sil)
        avg = silhouette_score(X_scaled, labels_for_sil)
        fig_sil, ax_s = plt.subplots(figsize=(6, 3))
        y_lower = 10
        clusters_to_plot = sorted([c for c in set(labels_for_sil) if c != -1]) + ([-1] if -1 in set(labels_for_sil) else [])
        for cl in clusters_to_plot:
            cl_vals = sil_vals[labels_for_sil == cl]
            cl_vals.sort()
            size = cl_vals.shape[0]
            if size == 0:
                continue
            y_upper = y_lower + size
            ax_s.fill_betweenx(np.arange(y_lower, y_upper), 0, cl_vals, alpha=0.7)
            ax_s.text(-0.05, y_lower + 0.5*size, str(cl))
            y_lower = y_upper + 10
        ax_s.axvline(x=avg, color="red", linestyle="--")
        ax_s.set_xlabel("Silhouette Coefficient")
        ax_s.set_yticks([])
        ax_s.set_title(f"Silhouette Plot — {recommended_algo} (avg={avg:.3f})")
        sil_png = fig_to_png_bytes(fig_sil)
        plt.close(fig_sil)
except Exception:
    sil_png = None

# 3) 2D PCA static scatter (matplotlib) for recommended algorithm
fig_pca2, axp = plt.subplots(figsize=(6, 4))
labels_plot = df[f"{recommended_algo}_Cluster"] if f"{recommended_algo}_Cluster" in df.columns else k_labels
unique_labels_plot = sorted(list(set(labels_plot)))
colors = plt.cm.get_cmap('tab10', max(3, len(unique_labels_plot)))
for i, lab in enumerate(unique_labels_plot):
    mask = labels_plot == lab
    axp.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(lab), s=40)
axp.set_xlabel("PC1")
axp.set_ylabel("PC2")
axp.set_title(f"2D PCA — {recommended_algo}")
axp.legend(title="cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
pca_png = fig_to_png_bytes(fig_pca2)
plt.close(fig_pca2)

# 4) Dendrogram PNG
try:
    Z_report = linkage(X_scaled, method=linkage_method)
    fig_den, axden = plt.subplots(figsize=(8, 3))
    dendrogram(Z_report, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=8, show_contracted=True)
    axden.set_title("Dendrogram")
    den_png = fig_to_png_bytes(fig_den)
    plt.close(fig_den)
except Exception:
    den_png = None

# Build PDF in memory
def build_pdf_bytes(recommendation_text, metrics_df, imgs_dict):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Smart Clustering Pro — Sample Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Recommendation", styles['Heading2']))
    story.append(Paragraph(recommendation_text, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Metrics Summary", styles['Heading2']))
    # Convert metrics_df to table data
    md = metrics_df.reset_index()
    table_data = [md.columns.tolist()] + md.values.tolist()
    story.append(Table(table_data))
    story.append(Spacer(1, 12))

    # Insert images
    for title, img_buf in imgs_dict.items():
        if img_buf is None:
            continue
        story.append(Paragraph(title, styles['Heading3']))
        img_buf.seek(0)
        # reportlab needs an actual file-like object; RLImage can accept a PIL object or filename
        # workaround: write to a temp file in working dir then embed
        tmp_name = f"tmp_{title.replace(' ','_')}.png"
        with open(tmp_name, 'wb') as f:
            f.write(img_buf.read())
        img = RLImage(tmp_name, width=400, height=200)
        story.append(img)
        story.append(Spacer(1, 12))
        # remove temp file after adding (can't remove until doc is built, so mark to delete later)
    doc.build(story)
    buf.seek(0)
    # cleanup temp files created
    for title in imgs_dict.keys():
        tmp_name = f"tmp_{title.replace(' ','_')}.png"
        if os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except Exception:
                pass
    return buf.getvalue()

# Prepare images dictionary
imgs = {
    "Elbow": elb_png,
    f"Silhouette_{recommended_algo}": sil_png,
    f"PCA2D_{recommended_algo}": pca_png,
    "Dendrogram": den_png
}

# Build recommendation text
rec_text = ai_recommend(pd.DataFrame(metrics).T)

# Build PDF bytes (on-demand, only build when user clicks)
if st.button("Generate PDF Report"):
    try:
        pdf_bytes = build_pdf_bytes(rec_text, pd.DataFrame(metrics).T, imgs)
        st.success("PDF report generated — ready to download")
        st.download_button("💾 Download PDF report", pdf_bytes, file_name="clustering_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")

# ---------------------------
# Final notes
# ---------------------------
st.markdown("""
---
**Notes & tips**
- Move the **K slider** to see K-Means clusters, silhouette and PCA change instantly.
- Use the **Elbow plot** to guide k selection visually.
- Use the **DBSCAN auto-scan** to find a good eps (it scans many candidates).
- HDBSCAN is optional — install with `pip install hdbscan` to enable it.
- For large datasets (>10k rows) consider sampling or increasing runtime resources.
""")
