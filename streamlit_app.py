
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


# Config
st.set_page_config(
    page_title="Lifestyle & Comorbidity Dashboard",
    layout="wide"
)

DATA_PATH = "processed_data.csv"

OUTCOME_COLS = [
    "Heart_Attack",
    "Diabetes",
    "Stroke",
    "Celiac_Disease",
    "Arthritis",
    "Liver_Disease",
    "Asthma",
    "COPD",
]

LIFESTYLE_COLS = [
    "Cigarettes_Per_Day",
    "Sleep_Hours",
    "Alcohol_Use_Frequency",
    "Physical_Activity_Equivalent_Min",
]

METABOLIC_COLS = [
    "BMI",
    "Waist_Circumference",
    "Total_Cholesterol",
    "HDL_Cholesterol",
    "LDL_Cholesterol",
    "Triglycerides",
    "Serum_Glucose",
    "Glycohemoglobin",
    "Platelet_Count",
    "Hemoglobin",
    "Hematocrit",
    "Iron_Saturation",
    "Serum_Iron",
    "Creatinine",
    "Uric_Acid",
    "WBC_Count",
    "Diastolic_BP_Average",
    "Systolic_BP_Average",
]

DEMOGRAPHIC_COLS = [
    "Age",
    "Gender",
    "Ethnicity",
    "Education_Level",
    "Income_to_Poverty_Ratio",
]

# def

def nice_label(name: str) -> str:
    return name.replace("_", " ")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Outcome - "Yes"/"No"
    for col in OUTCOME_COLS:
        df[col] = df[col].astype(str)

    # Create comorbidity summary variables
    outcome_yes = df[OUTCOME_COLS].apply(lambda c: c.str.upper().eq("YES"))
    df["Comorbidity_Count"] = outcome_yes.sum(axis=1)
    df["Any_Comorbidity"] = np.where(df["Comorbidity_Count"] > 0, "Yes", "No")

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Demographics 
    st.sidebar.subheader("Demographics")

    # Age
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider(
        "Age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        step=1,
    )

    # Gender
    genders = sorted(df["Gender"].dropna().unique())
    selected_genders = st.sidebar.multiselect(
        "Gender",
        options=genders,
        default=genders,
    )

    # Ethnicity
    ethnicities = sorted(df["Ethnicity"].dropna().unique())
    selected_ethnicities = st.sidebar.multiselect(
        "Ethnicity",
        options=ethnicities,
        default=ethnicities,
    )

    # Education
    edu_levels = sorted(df["Education_Level"].dropna().unique())
    selected_edu = st.sidebar.multiselect(
        "Education level",
        options=edu_levels,
        default=edu_levels,
    )

    # Income-to-poverty ratio
    inc_min, inc_max = float(df["Income_to_Poverty_Ratio"].min()), float(df["Income_to_Poverty_Ratio"].max())
    income_range = st.sidebar.slider(
        "Income-to-poverty ratio",
        min_value=float(round(inc_min, 1)),
        max_value=float(round(inc_max, 1)),
        value=(float(round(inc_min, 1)), float(round(inc_max, 1))),
        step=0.1,
    )

    # Lifestyle 
    st.sidebar.subheader("Lifestyle")

    # Cigarettes per day
    c_min, c_max = int(df["Cigarettes_Per_Day"].min()), int(df["Cigarettes_Per_Day"].max())
    cig_range = st.sidebar.slider(
        "Cigarettes per day",
        min_value=c_min,
        max_value=c_max,
        value=(c_min, c_max),
        step=1,
    )

    # Sleep hours
    s_min, s_max = int(df["Sleep_Hours"].min()), int(df["Sleep_Hours"].max())
    sleep_range = st.sidebar.slider(
        "Sleep hours",
        min_value=s_min,
        max_value=s_max,
        value=(s_min, s_max),
        step=1,
    )

    # Alcohol use frequency (numeric code; you can later map to labels)
    a_min, a_max = int(df["Alcohol_Use_Frequency"].min()), int(df["Alcohol_Use_Frequency"].max())
    alcohol_range = st.sidebar.slider(
        "Alcohol use frequency (code)",
        min_value=a_min,
        max_value=a_max,
        value=(a_min, a_max),
        step=1,
    )

    # Physical activity
    p_min, p_max = int(df["Physical_Activity_Equivalent_Min"].min()), int(df["Physical_Activity_Equivalent_Min"].max())
    pa_range = st.sidebar.slider(
        "Physical activity (equivalent minutes per week)",
        min_value=p_min,
        max_value=p_max,
        value=(p_min, p_max),
        step=10,
    )

    # Build mask
    mask = (
        df["Age"].between(age_range[0], age_range[1])
        & df["Income_to_Poverty_Ratio"].between(income_range[0], income_range[1])
        & df["Cigarettes_Per_Day"].between(cig_range[0], cig_range[1])
        & df["Sleep_Hours"].between(sleep_range[0], sleep_range[1])
        & df["Alcohol_Use_Frequency"].between(alcohol_range[0], alcohol_range[1])
        & df["Physical_Activity_Equivalent_Min"].between(pa_range[0], pa_range[1])
    )

    if selected_genders:
        mask &= df["Gender"].isin(selected_genders)
    if selected_ethnicities:
        mask &= df["Ethnicity"].isin(selected_ethnicities)
    if selected_edu:
        mask &= df["Education_Level"].isin(selected_edu)

    return df[mask].copy()


def prevalence_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in OUTCOME_COLS:
        s = df[col].str.upper()
        prev = (s == "YES").mean() * 100
        rows.append(
            {"Outcome": nice_label(col), "Prevalence (%)": round(prev, 1)}
        )
    prev_df = pd.DataFrame(rows)
    return prev_df.sort_values("Prevalence (%)", ascending=False)


# Main app

df = load_data(DATA_PATH)

st.title("Lifestyle & Comorbidity Explorer")
st.markdown(
    """
This dashboard lets you interactively explore how **lifestyle behaviours**  
(smoking, sleep, physical activity, alcohol) and **demographics** relate to:

- Metabolic and cardiometabolic markers (BMI, lipids, glucose, blood pressure, etc.)
- Prevalence and burden of chronic conditions (heart attack, diabetes, stroke, COPD, etc.)
"""
)

filtered = apply_filters(df)

st.markdown(f"### Current selection: {len(filtered):,} participants")

if filtered.empty:
    st.warning("No participants match the current filter settings. Try relaxing your filters.")
    st.stop()



# 1. Disease prevalence summary

st.subheader("Comorbidity overview under current filters")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "Any comorbidity (≥1 condition)",
        f"{(filtered['Any_Comorbidity'] == 'Yes').mean() * 100:.1f} %",
    )
with col2:
    st.metric(
        "Mean comorbidity count",
        f"{filtered['Comorbidity_Count'].mean():.2f}",
    )
with col3:
    st.metric(
        "Max comorbidity count",
        int(filtered["Comorbidity_Count"].max()),
    )

prev_df = prevalence_table(filtered)

prev_chart = (
    alt.Chart(prev_df)
    .mark_bar()
    .encode(
        x=alt.X("Prevalence (%):Q", title="Prevalence (%)"),
        y=alt.Y("Outcome:N", sort="-x", title=""),
        tooltip=["Outcome", "Prevalence (%)"],
    )
    .properties(height=300)
)

st.altair_chart(prev_chart, use_container_width=True)



# 2. Metabolic markers across comorbidity levels

st.subheader("Metabolic markers across comorbidity levels")

cm_y_var = st.selectbox(
    "Metabolic variable (y-axis, by comorbidity count)",
    options=METABOLIC_COLS,
    format_func=nice_label,
    index=METABOLIC_COLS.index("BMI") if "BMI" in METABOLIC_COLS else 0,
)

# 箱线图 + jitter 点
box = (
    alt.Chart(filtered)
    .mark_boxplot()
    .encode(
        x=alt.X("Comorbidity_Count:O",
                title="Number of comorbid conditions"),
        y=alt.Y(cm_y_var, title=nice_label(cm_y_var)),
        color=alt.Color("Comorbidity_Count:O", legend=None),
        tooltip=["Comorbidity_Count:O", cm_y_var]
    )
    .properties(height=350)
)

points = (
    alt.Chart(filtered)
    .mark_circle(size=20, opacity=0.3)
    .encode(
        x=alt.X("Comorbidity_Count:O"),
        y=alt.Y(cm_y_var),
        color=alt.Color("Comorbidity_Count:O", legend=None),
    )
)

st.altair_chart(box + points, use_container_width=True)


# 3. Comorbidity prevalence by lifestyle level

st.subheader("Comorbidity prevalence by lifestyle level")

life_var = st.selectbox(
    "Lifestyle variable to group by",
    options=LIFESTYLE_COLS,
    format_func=nice_label,
    index=LIFESTYLE_COLS.index("Physical_Activity_Equivalent_Min")
      if "Physical_Activity_Equivalent_Min" in LIFESTYLE_COLS else 0,
)

# 用分位数把生活方式变量切成 3 组：Low / Medium / High
q = filtered[life_var].quantile([0, 1/3, 2/3, 1]).to_list()
labels = ["Low", "Medium", "High"]

filtered_life = filtered.copy()
filtered_life["Lifestyle_Group"] = pd.cut(
    filtered_life[life_var],
    bins=q,
    labels=labels,
    include_lowest=True,
    duplicates="drop"
)

# 计算每组的任意共病患病率
group_prev = (
    filtered_life
    .groupby("Lifestyle_Group", dropna=True)
    .apply(lambda d: (d["Any_Comorbidity"] == "Yes").mean())
    .reset_index(name="Prevalence")
)

prev_bar = (
    alt.Chart(group_prev)
    .mark_bar()
    .encode(
        x=alt.X("Lifestyle_Group:N",
                title=f"{nice_label(life_var)} level"),
        y=alt.Y("Prevalence:Q",
                title="Any comorbidity prevalence",
                axis=alt.Axis(format=".0%")),
        tooltip=[
            alt.Tooltip("Lifestyle_Group:N", title="Group"),
            alt.Tooltip("Prevalence:Q", format=".1%", title="Prevalence"),
        ]
    )
    .properties(height=300)
)

st.altair_chart(prev_bar, use_container_width=True)



# 4. Correlation heatmap (metabolic block)

st.subheader("Correlation of metabolic measures (current filters)")

corr_df = filtered[METABOLIC_COLS].corr().stack().reset_index()
corr_df.columns = ["Var1", "Var2", "Correlation"]

heatmap = (
    alt.Chart(corr_df)
    .mark_rect()
    .encode(
        x=alt.X("Var1:N", title="", sort=METABOLIC_COLS),
        y=alt.Y("Var2:N", title="", sort=METABOLIC_COLS),
        color=alt.Color("Correlation:Q", scale=alt.Scale(scheme="redblue"), title="r"),
        tooltip=["Var1", "Var2", alt.Tooltip("Correlation:Q", format=".2f")],
    )
    .properties(height=400)
)

st.altair_chart(heatmap, use_container_width=True)