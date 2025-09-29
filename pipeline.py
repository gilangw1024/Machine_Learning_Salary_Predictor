import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def clean_company_size(val):
    if pd.isna(val):
        return None
    val = str(val).lower().strip().replace('$', '').replace(',', '')
    if '-' in val:
        try:
            low, high = val.split('-')
            return (float(low) + float(high)) / 2
        except:
            return None
    if val in ['private', 'unknown', '—', 'n/a']:
        return None
    try:
        return float(val)
    except:
        return None

def clean_salary(val):
    if pd.isna(val):
        return None
    val = str(val).lower().replace('€', '').replace(',', '').strip()
    if '-' in val:
        try:
            low, high = val.split('-')
            return (float(low.strip()) + float(high.strip())) / 2
        except:
            return None
    try:
        return float(val)
    except:
        return None

def safe_eval(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except:
        return []

def preprocess_for_training(df):
    df['company_size_clean'] = df['company_size'].apply(clean_company_size)
    median_size = df['company_size_clean'].median()
    df['company_size_clean'] = df['company_size_clean'].fillna(median_size)

    df['salary_clean'] = df['salary'].apply(clean_salary)
    upper_limit = df['salary_clean'].quantile(0.99)
    df = df[df['salary_clean'] <= upper_limit]

    df['job_title'] = df['job_title'].fillna('data scientist')
    df['seniority_level'] = df['seniority_level'].fillna(df['seniority_level'].mode()[0])
    df['status'] = df['status'].fillna('unknown')
    df['ownership'] = df['ownership'].fillna('unknown')

    df['skills'] = df['skills'].apply(safe_eval)
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df['skills'])
    skills_df = pd.DataFrame(skills_encoded, columns=[f"skill_{s}" for s in mlb.classes_], index=df.index)

    df = pd.concat([df.drop(columns=['skills']), skills_df], axis=1)
    df = pd.get_dummies(df, columns=['status', 'industry', 'ownership', 'job_title'], prefix=['status', 'industry', 'ownership', 'job'])
    df['seniority_level_encoded'] = pd.Categorical(
        df['seniority_level'].str.strip().str.lower(),
        categories=['junior', 'midlevel', 'senior', 'lead'],
        ordered=True
    ).codes.astype(float)
    df.drop(columns=['seniority_level', 'company_size', 'salary', 'post_date', 'revenue', 'location', 'headquarter', 'company'], inplace=True)

    X = df.drop(columns=['salary_clean'])
    y = df['salary_clean']
    return X, y, mlb  # mlb disimpan untuk deployment

def preprocess_for_prediction(df, mlb):
    df['company_size_clean'] = df['company_size'].apply(clean_company_size)
    median_size = df['company_size_clean'].median()
    df['company_size_clean'] = df['company_size_clean'].fillna(median_size)

    df['job_title'] = df['job_title'].fillna('data scientist')
    df['seniority_level'] = df['seniority_level'].fillna('midlevel')
    df['status'] = df['status'].fillna('unknown')
    df['ownership'] = df['ownership'].fillna('unknown')

    df['skills'] = df['skills'].apply(safe_eval)
    skills_encoded = mlb.transform(df['skills'])
    skills_df = pd.DataFrame(skills_encoded, columns=[f"skill_{s}" for s in mlb.classes_], index=df.index)

    df = pd.concat([df.drop(columns=['skills']), skills_df], axis=1)
    df = pd.get_dummies(df, columns=['status', 'industry', 'ownership', 'job_title'], prefix=['status', 'industry', 'ownership', 'job'])
    df['seniority_level_encoded'] = pd.Categorical(
        df['seniority_level'].str.strip().str.lower(),
        categories=['junior', 'midlevel', 'senior', 'lead'],
        ordered=True
    ).codes.astype(float)
    df.drop(columns=['seniority_level', 'company_size', 'post_date', 'revenue', 'location', 'headquarter', 'company'], inplace=True)

    return df
