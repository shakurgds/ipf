import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Project Prioritization", layout="wide")

# --- SIDEBAR: Global Settings ---
st.sidebar.title("Settings")

num_projects = st.sidebar.number_input("Number of Projects", min_value=2, max_value=10, value=3)
budget = st.sidebar.number_input("Total Budget", min_value=0.0, value=100.0)

st.sidebar.markdown("### SEI Criteria")
sei_criteria = st.sidebar.text_area("SEI Criteria (comma-separated)", "Biodiversity, Social Impact")
sei_criteria = [c.strip() for c in sei_criteria.split(",")]

st.sidebar.markdown("### FEI Criteria")
fei_criteria = st.sidebar.text_area("FEI Criteria (comma-separated)", "ROI, Payback Period")
fei_criteria = [c.strip() for c in fei_criteria.split(",")]

st.sidebar.markdown("### IPF Weights")

# First slider: SEI weight (0 to 1)
w_sei = st.sidebar.slider("Weight for SEI", 0.0, 1.0, 0.5, step=0.01)
w_fei = 1.0 - w_sei
st.sidebar.write(f"Weight for FEI: {w_fei:.2f}")

st.sidebar.markdown("### Budget Limits (Quadrant Lines)")
sei_budget_threshold = st.sidebar.number_input("SEI Budget Limit (Y axis)", min_value=0.0, value=45.0)
fei_budget_threshold = st.sidebar.number_input("FEI Budget Limit (X axis)", min_value=0.0, value=69.0)

# --- MAIN: Project Data Entry ---
st.title("Project Prioritization Tool")

projects = []  # Collect project data here
project_tabs = st.tabs([f"Project {i+1}" for i in range(num_projects)])
for i, tab in enumerate(project_tabs):
    with tab:
        st.markdown(f"### {i+1}")
        name = st.text_input(f"Name", value=f"Project {i+1}", key=f"name_{i}")
        cost = st.number_input(f"Cost", min_value=0.0, key=f"cost_{i}")
        sei_col, fei_col = st.columns(2)
        with sei_col:
            st.markdown("**SEI Scores**")
            sei_scores = [st.slider(f"{crit}", 0, 10, 5, key=f"sei_{i}_{j}", help=f"Score for {crit}") for j, crit in enumerate(sei_criteria)]
        with fei_col:
            st.markdown("**FEI Scores**")
            fei_scores = [st.slider(f"{crit}", 0, 10, 5, key=f"fei_{i}_{j}", help=f"Score for {crit}") for j, crit in enumerate(fei_criteria)]
        projects.append({
            "Name": name,
            "Cost": cost,
            "SEI Scores": sei_scores,
            "FEI Scores": fei_scores
        })

# --- PCA and Normalization ---
def min_max_scale(arr):
    z_min = np.min(arr)
    z_max = np.max(arr)
    if z_max == z_min:
        return np.full_like(arr, 50.0)
    return (arr - z_min) / (z_max - z_min) * 100

# PCA for SEI
sei_matrix = np.array([p["SEI Scores"] for p in projects])
if sei_matrix.shape[0] > 1:
    sei_pca = PCA(n_components=1).fit_transform(sei_matrix)
    for idx, p in enumerate(projects):
        p["SEI"] = sei_pca[idx, 0]
else:
    for p in projects:
        p["SEI"] = np.mean(p["SEI Scores"])
sei_scaled = min_max_scale(np.array([p["SEI"] for p in projects]))
for idx, p in enumerate(projects):
    p["SEI"] = sei_scaled[idx]

# PCA for FEI
fei_matrix = np.array([p["FEI Scores"] for p in projects])
if fei_matrix.shape[0] > 1:
    fei_pca = PCA(n_components=1).fit_transform(fei_matrix)
    for idx, p in enumerate(projects):
        p["FEI"] = fei_pca[idx, 0]
else:
    for p in projects:
        p["FEI"] = np.mean(p["FEI Scores"])
fei_scaled = min_max_scale(np.array([p["FEI"] for p in projects]))
for idx, p in enumerate(projects):
    p["FEI"] = fei_scaled[idx]

# IPF Calculation
for p in projects:
    p["IPF"] = w_sei * p["SEI"] + w_fei * p["FEI"]

df = pd.DataFrame([{
    "Name": p["Name"],
    "Cost": p["Cost"],
    "SEI": p["SEI"],
    "FEI": p["FEI"],
    "IPF": p["IPF"]
} for p in projects])

# Sort by IPF, select projects within budget
df_sorted = df.sort_values("IPF", ascending=False).reset_index(drop=True)
df_sorted["CumulativeCost"] = df_sorted["Cost"].cumsum()
df_sorted["Selected"] = df_sorted["CumulativeCost"] <= budget

df_ipf = df.sort_values("IPF", ascending=False).reset_index(drop=True)

# Calculate cumulative cost and within-budget status for IPF ranking
df_ipf["CumulativeCost"] = df_ipf["Cost"].cumsum()
df_ipf["WithinBudget"] = df_ipf["CumulativeCost"] <= budget

# Create a new bar chart with a single color scheme
fig_ipf = px.bar(
    df_ipf,
    x="Name",
    y="IPF",
    title="Project Prioritization by IPF (Combined SEI/FEI Ranking)",
    labels={"IPF": "Integrated Prioritization Function (IPF)"},
    color="IPF",  # This will be used for the opacity
    color_continuous_scale=[(0, "rgba(31, 119, 180, 0.2)"), (1, "rgba(31, 119, 180, 1)")],  # Blue color with varying opacity
)

# Update layout for better visualization
fig_ipf.update_layout(
    xaxis_title="Project",
    yaxis_title="IPF Score",
    showlegend=False,  # Remove legend since we're using a single color
    plot_bgcolor="white",
    bargap=0.3,  # Add some space between bars
    coloraxis_colorbar=dict(
        title="IPF",
        orientation='h',
        x=0.5,  # Centered horizontally
        y=1.08, # Just above the chart
        xanchor='center',
        len=0.5,  # Length of the colorbar
        thickness=15,
        tickvals=[df_ipf["IPF"].min(), df_ipf["IPF"].max()],
        ticktext=["Lower", "Higher"]
    )
)

# Add value labels on top of bars
fig_ipf.update_traces(
    texttemplate="%{y:.1f}",
    textposition="outside",
    marker_line_width=0,  # Remove bar borders
)

# Find the index of the last project within budget
budget_limit_index = df_ipf[df_ipf["CumulativeCost"] <= budget].index.max()

# Add vertical dashed line for budget limit
fig_ipf.add_vline(
    x=budget_limit_index + 0.5,
    line_dash="dash",
    line_color="brown",
    annotation_text="budget limit",
    annotation_position="top right"
)

st.plotly_chart(fig_ipf, use_container_width=True)

# Display budget information under the bar chart
st.write(f"**Total Available Budget:** ${budget:,.2f}")
st.write(f"**Number of Projects Within Budget:** {budget_limit_index + 1 if budget_limit_index >= 0 else 0}")
if budget_limit_index >= 0:
    st.write(f"**Total Budget of Selected Projects:** ${df_ipf['CumulativeCost'].iloc[budget_limit_index]:,.2f}")
else:
    st.write("**Total Budget of Selected Projects:** $0.00")

# Add a note about the color intensity
st.caption("Note: Color intensity indicates IPF score - darker blue represents higher priority.")

# Plot: color by selection
fig = px.scatter(
    df,
    x="FEI",
    y="SEI",
    text="Name",
    color="Name",
    title="Projects: SEI vs FEI",
    labels={
        "FEI": "Financial/Economic Index (FEI, 0–100)",
        "SEI": "Social/Environmental Index (SEI, 0–100)",
        "Name": "Project"
    }
)

# Set all markers to the same size without outlines and position labels
fig.update_traces(
    marker=dict(size=16),
    textposition="top right",
    textfont=dict(size=12)
)

# Add budget threshold lines
fig.add_vline(
    x=fei_budget_threshold,
    line_dash="dash",
    line_color="red",
    annotation_text="FEI Budget Limit",
    annotation_position="top right"
)
fig.add_hline(
    y=sei_budget_threshold,
    line_dash="dash",
    line_color="blue",
    annotation_text="SEI Budget Limit",
    annotation_position="bottom left"
)

# Calculate margins for axis ranges
x_min = df['FEI'].min()
x_max = df['FEI'].max()
y_min = df['SEI'].min()
y_max = df['SEI'].max()

x_margin = (x_max - x_min) * 0.1 if x_max > x_min else 10
y_margin = (y_max - y_min) * 0.1 if y_max > y_min else 10

# Use the axis range (with margins) to calculate quadrant centers
x_left = x_min - x_margin
x_right = x_max + x_margin
y_bottom = y_min - y_margin
y_top = y_max + y_margin

def lerp(a, b, t):
    return a + (b - a) * t

# 75% towards the edge for A/B (top), 25% for C/D (bottom)
A_xc = lerp(fei_budget_threshold, x_right, 0.75)
A_yc = lerp(sei_budget_threshold, y_top, 0.75)

B_xc = lerp(x_left, fei_budget_threshold, 0.25)
B_yc = lerp(sei_budget_threshold, y_top, 0.75)

C_xc = lerp(fei_budget_threshold, x_right, 0.75)
C_yc = lerp(y_bottom, sei_budget_threshold, 0.25)

D_xc = lerp(x_left, fei_budget_threshold, 0.25)
D_yc = lerp(y_bottom, sei_budget_threshold, 0.25)

fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05,
        title="Projects",
        itemsizing="constant"
    ),
    annotations=[
        #dict(x=A_xc, y=A_yc, xref="x", yref="y", text="A", showarrow=False, font=dict(size=16, color="black"),
             #bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, borderpad=2),
        #dict(x=B_xc, y=B_yc, xref="x", yref="y", text="B", showarrow=False, font=dict(size=16, color="black"),
             #bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, borderpad=2),
        #dict(x=C_xc, y=C_yc, xref="x", yref="y", text="C", showarrow=False, font=dict(size=16, color="black"),
             #bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, borderpad=2),
        #dict(x=D_xc, y=D_yc, xref="x", yref="y", text="D", showarrow=False, font=dict(size=16, color="black"),
             #bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1, borderpad=2),
    ],
    margin=dict(r=220),
    xaxis_range=[x_min - x_margin, x_max + x_margin],
    yaxis_range=[y_min - y_margin, y_max + y_margin]
)

st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Dashed red line: FEI budget constraint. Dashed blue line: SEI budget constraint. "
    "Projects to the right of the red line or above the blue line are beyond the budget envelope for that axis."
)

# Add two-column legend for quadrant labels below the chart
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; gap: 2em; font-weight: bold;'>
        <span>A: Higher Priority (Upper Right)</span>
        <span>B: Higher Social/Environmental Priority (Upper Left)</span>
        <span>C: Higher Financial/Economic Priority (Lower Right)</span>
        <span>D: Lower Priority (Lower Left)</span>
    </div>
    """,
    unsafe_allow_html=True
)

