import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"data berhasil di-load: {df.shape}")
    return df

def preprocess(df):
    # hapus kolom tidak diperlukan
    df_clean = df.drop(columns=['StudentID'])

    # pisahkan fitur dan target
    X = df_clean.drop(columns=['GradeClass'])
    y = df_clean['GradeClass'].astype(int)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scaling fitur numerik
    scaler = StandardScaler()
    num_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler

def save_results(X_train, X_test, y_train, y_test, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X_train_out = X_train.copy()
    X_train_out['GradeClass'] = y_train.values
    X_train_out.to_csv(f'{output_dir}/train.csv', index=False)

    X_test_out = X_test.copy()
    X_test_out['GradeClass'] = y_test.values
    X_test_out.to_csv(f'{output_dir}/test.csv', index=False)

    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"semua file tersimpan di: {output_dir}")

if __name__ == '__main__':
    df = load_data('student_performance_raw.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    save_results(X_train, X_test, y_train, y_test, scaler, 'student_performance_preprocessing')
    print("preprocessing selesai!")