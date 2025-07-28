"""
Exploratory Data Analysis (EDA) Script for:
"Early Detection of Heart Disease: A Machine Learning Approach Using CDC Health Indicators (2022)"
Author: Mariam Sitoe Lo
Date: June 16, 2025
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder

# Configure display and aesthetics
pd.set_option('display.max_columns', None)
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# ========================== Load Data ==========================
def load_data(filepath):
    """Loads the dataset from the given file path."""
    return pd.read_csv(filepath)

def ensure_folder_exists(folder_path):
    """Creates the folder if it doesn't exist."""
    os.makedirs(folder_path, exist_ok=True)

# ========================== Basic Overview of Data Structure ==========================
def summarize_data(df, output_folder):
    """Prints basic information and saves summary stats to CSV."""
    print("\n========================== Basic Overview of Data Structure ==========================")
    print("Df.head(): \n", df.head())
    print("Df.shape: ", df.shape)
    print("Null Values: \n", df.isnull().sum())
    print("Columns: ", df.columns)
    print("Duplicates: ", df.duplicated().sum())
    print("Summary Statistics: \n", df.describe())
    #df.describe().to_csv(os.path.join(output_folder, 'summary_statistics.csv'))
    print("\nTarget Class Distribution:")
    print(df['HadHeartAttack'].value_counts(normalize=True))
    df.describe().T.style.set_properties(**{'background-color': 'grey','color': 'white','border-color': 'white'})



# ========================== Clean Column Names ==========================
def clean_column_names(df):
    """Standardizes column names by removing special characters and spaces."""
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "").str.replace("/", "_")
    print("Cleaned column names: ", df.columns)
    return df

# ========================== Univariate Analysis ==========================
# def normalized_barplot(df, x_col, hue_col, output_folder, filename):
#     """Creates and saves a normalized stacked bar plot."""
#     ct = pd.crosstab(df[x_col], df[hue_col], normalize='index')
#     ct.plot(kind='bar', stacked=True, colormap='autumn', figsize=(10, 6))
#     plt.ylabel('Proportion')
#     plt.title(f'Proportion of {hue_col} by {x_col}')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_folder, filename))
#     plt.close()

def univariate_analysis(df, output_folder):
    """Generates and saves plots for general health, age, BMI, sleep hours, and target variable."""
    print("\n===== Univariate Analysis =====")

    for col in ['GeneralHealth', 'AgeCategory', 'HadHeartAttack']:
        #shows why we need smote
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, data=df, color='orange',order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{col.lower()}_distribution.png'))
        plt.close()

    # BMI Distribution
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.kdeplot(df[df["HadHeartAttack"] == "Yes"]["BMI"], fill=True, alpha=0.5, color="red", label="HadHeartAttack", ax=ax)
    sns.kdeplot(df[df["HadHeartAttack"] == "No"]["BMI"], fill=True, alpha=0.5, color="#f1a41e", label="No Heart Attack", ax=ax)
    ax.set_title('Distribution of Body Mass Index by Heart Attack History', fontsize=16)
    ax.set_xlabel("BMI")
    ax.set_ylabel("Density")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'bmi_kde_distribution.png'))
    plt.close()

    # Sleep Hours Distribution
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.kdeplot(df[df["HadHeartAttack"] == "Yes"]["SleepHours"], fill=True, alpha=0.5, color="red", label="HadHeartAttack", ax=ax)
    sns.kdeplot(df[df["HadHeartAttack"] == "No"]["SleepHours"], fill=True, alpha=0.5, color="#f1a41e", label="No Heart Attack", ax=ax)
    ax.set_title('Distribution of Sleep Hours by Heart Attack History', fontsize=16)
    ax.set_xlabel("Sleep Hours")
    ax.set_ylabel("Density")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'sleeptime_kde_distribution.png'))
    plt.close()

    # Physical Health Days Distribution
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.kdeplot(df[df["HadHeartAttack"] == "Yes"]["PhysicalHealthDays"], fill=True, alpha=0.5, color="red", label="HadHeartAttack", ax=ax)
    sns.kdeplot(df[df["HadHeartAttack"] == "No"]["PhysicalHealthDays"], fill=True, alpha=0.5, color="#f1a41e", label="No Heart Attack", ax=ax)
    ax.set_title('Distribution of Physical Health (last 30 days)', fontsize=16)
    ax.set_xlabel("Physical Health Days")
    ax.set_ylabel("Density")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'physicalhealth_kde_distribution.png'))
    plt.close()

    # Mental Health Days Distribution
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.kdeplot(df[df["HadHeartAttack"] == "Yes"]["MentalHealthDays"], fill=True, alpha=0.5, color="red", label="HadHeartAttack", ax=ax)
    sns.kdeplot(df[df["HadHeartAttack"] == "No"]["MentalHealthDays"], fill=True, alpha=0.5, color="#f1a41e", label="No Heart Attack", ax=ax)
    ax.set_title('Distribution of Mental Health (last 30 days)', fontsize=16)
    ax.set_xlabel("Mental Health Days")
    ax.set_ylabel("Density")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'mentalhealth_kde_distribution.png'))
    plt.close()


    # Histograms and Boxplots for numerical variables
    for col in ['BMI', 'SleepHours', 'PhysicalHealthDays', 'MentalHealthDays']:
        # plt.figure(figsize=(8, 4))
        # sns.histplot(df[col].dropna(), bins=30, kde=True,color='orange')
        # plt.title(f"Histogram of {col}")
        # plt.tight_layout()
        # plt.savefig(os.path.join(output_folder, f'{col.lower()}_hist.png'))
        # plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col],color='orange')
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{col.lower()}_boxplot.png'))
        plt.close()

    # Pairplot for select features
    # sns.pairplot(df[['BMI', 'SleepHours', 'PhysicalHealthDays', 'MentalHealthDays', 'HadHeartAttack']], hue='HadHeartAttack', diag_kind='kde', plot_kws={'alpha': 0.5})
    # plt.savefig(os.path.join(output_folder, 'pairplot_selected_features.png'))
    # plt.close()

# ========================== Bivariate Analysis ==========================
def bivariate_analysis(df, output_folder):
    """Analyzes relationships between features and heart attack history."""
    print("\n===== Bivariate Analysis =====")

    # Heart Attack by Sex
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Sex', hue='HadHeartAttack',palette='autumn', data=df)
    plt.title("Heart Attack History by Sex")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'heart_attack_by_sex.png'))
    plt.close()

    # Heart Attack by Age Group 
    plt.figure(figsize=(13, 10))
    sns.countplot(x='AgeCategory', hue='HadHeartAttack', data=df, palette='autumn')
    plt.title("Heart Attack Proportion by Age Group")
    plt.xlabel('AgeCategory')
    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.legend(['Normal', 'HeartDisease'])
    plt.tight_layout() 
    plt.subplots_adjust(top=0.9, bottom=0.15)  
    plt.savefig(os.path.join(output_folder, 'heart_attack_by_age_group.png'))
    plt.close()



    # Heart Attack by Race
    plt.figure(figsize = (13,6))
    sns.countplot( x= df['RaceEthnicityCategory'], hue = 'HadHeartAttack', data = df,order=df['RaceEthnicityCategory'].value_counts().index, palette = 'autumn')
    plt.xlabel('Race')
    plt.legend()
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'heart_attack_by_race.png'))
    plt.close()

    # Scatter Plots
    # scatter_pairs = [('BMI', 'SleepHours'), ('BMI', 'PhysicalHealthDays'), ('SleepHours', 'MentalHealthDays')]
    # for x, y in scatter_pairs:
    #     plt.figure(figsize=(6, 4))
    #     sns.scatterplot(x=x, y=y, data=df, hue='HadHeartAttack', alpha=0.5)
    #     plt.title(f"{x} vs {y} (colored by Heart Attack History)")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_folder, f'scatter_{x.lower()}_vs_{y.lower()}.png'))
    #     plt.close()

    # Chi-square tests
    print("\n===== Chi-Square Tests =====")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if col == 'HadHeartAttack' or df[col].nunique() > 10:
            continue
        contingency = pd.crosstab(df[col], df['HadHeartAttack'])
        chi2, p, _, _ = chi2_contingency(contingency)
        print(f"{col} vs HadHeartAttack: χ² = {chi2:.2f}, p = {p:.4f}")

# ========================== Correlation Analysis ==========================
def correlation_analysis(df, output_folder):
    """Plots and saves correlation heatmap """

    selected_features = [
    'SleepHours', 'BMI', 'PhysicalHealthDays', 'MentalHealthDays',
    'HadHeartAttack', 'Sex', 'AgeCategory', 'GeneralHealth','HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadKidneyDisease', 
       'HadDiabetes','DifficultyWalking','AlcoholDrinkers','SmokerStatus','ECigaretteUsage'
]


    df_selected = df[selected_features].copy()
    for col in df_selected.select_dtypes(include='object').columns:
        df_selected[col] = LabelEncoder().fit_transform(df_selected[col])

    correlation = df_selected.corr().round(2)
    print(correlation)
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='YlOrBr', annot_kws={"size": 9})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Correlation Matrix: Selected Features", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'correlation_selected_features.png'))
    plt.close()


# ========================== Categorical Summary ==========================
def categorical_summary(df,output_folder):
    """Prints unique value counts of categorical columns."""
    print("\n===== Categorical Variable Summary =====")
    nunique = df.nunique().sort_values(ascending=False)
    print(nunique)
    #nunique.to_csv(os.path.join(output_folder, 'nunique_summary.csv'))
    print("\nSuggested categorical features for encoding:")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        print(f"{col}: {df[col].nunique()} unique values")

# ========================== Group Statistics ==========================
def group_stats(df, output_folder):
    """Prints and saves mean BMI and sleep hours by general health."""
    print("\n📈 Average Sleep Hours and BMI by General Health:")
    group = df.groupby("GeneralHealth")[["SleepHours", "BMI"]].mean().sort_values("SleepHours", ascending=False)
    print(group)
    #group.to_csv(os.path.join(output_folder, 'group_stats_by_general_health.csv'))

# ========================== Main ==========================
def main():
    data_path = r"C:\\Users\\lomar\\Documents\\Mariam - Master's\\MRP\\heart_2022_no_nans.csv"
    output_folder = r"C:\\Users\\lomar\\Documents\\Mariam - Master's\\MRP\\EDA"
    ensure_folder_exists(output_folder)
    df = load_data(data_path)

    summarize_data(df, output_folder)
    df = clean_column_names(df)
    univariate_analysis(df, output_folder)
    #normalized_barplot(df, 'AgeCategory', 'HadHeartAttack', output_folder, 'normalized_heart_attack_by_age.png')
    #normalized_barplot(df, 'RaceEthnicityCategory', 'HadHeartAttack', output_folder, 'normalized_heart_attack_by_race.png')

    bivariate_analysis(df, output_folder)
    correlation_analysis(df, output_folder)
    categorical_summary(df,output_folder)
    group_stats(df, output_folder)
    print("\nEDA complete.")

if __name__ == "__main__":
    main()
