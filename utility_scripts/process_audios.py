import os
from pydub import AudioSegment
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance

scam_chunks = []
ham_chunks = []

# Initialize wav2vec model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()  # Set to evaluation mode


def chunk_audio(type, df=None):
    if df is None:
        df = pd.DataFrame([])
    source_dir = f"data/audios/{type}"
    chunk_length = 2
    silence_threshold = 0.0004

    for root, _, files in os.walk(source_dir):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
                
            y, sr = librosa.load(file_path, sr=None)
            segment_samples = int(chunk_length * sr)
    
            # Split the audio into segments
            segments = []

            for start in range(0, len(y), segment_samples):
                end = min(start + segment_samples, len(y))
                segment = y[start:end]
                if len(segment) >= segment_samples * 0.5 and librosa.feature.rms(y=segment).max() > silence_threshold:
                    segments.append(segment)

            for i in range(len(segments)):
                chunk = segments[i]
                segment_features = {}
                
                if sr != 16000:
                    chunk = librosa.resample(chunk, orig_sr=sr, target_sr=16000)

                segment_features['samples'] = json.dumps(chunk.tolist()) 

                # mfccs = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=NUM_MFCCS)
                    
                # # Extract spectral features
                # spectral_centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)
                # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sr)
                # spectral_rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr)

                # segment_features = {'mfccs_' + str(i): np.mean(mfccs[i]) for i in range(NUM_MFCCS)}
                # segment_features['spectral_centroid'] = np.mean(spectral_centroid)
                # segment_features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
                # segment_features['spectral_rolloff'] = np.mean(spectral_rolloff)
                # segment_features['rms'] = librosa.feature.rms(y=chunk).max()
                segment_features['source'] = file
                if type == "scam":
                    segment_features['label'] = 0
                else:
                    segment_features['label'] = 1
                df = pd.concat([df, pd.DataFrame([segment_features])], ignore_index=True)
                print(df.shape)
    return df

def train_evaluate_model(X_train, X_test, y_train, y_test, model_name, model):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    spam_recall = report['0']['recall']
    
    print(f"\n{model_name} Results:")
    print(f"Overal Accuracy: {accuracy:.4f}")
    print(f"% of Spam Caught: {spam_recall:.4f}")
    
    return model

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)

    print(sum(y_pred) / len(y_pred))

if __name__ == "__main__":
    df = pd.DataFrame([])
    df = chunk_audio("scam", df)
    df = chunk_audio("ham", df)
    df.to_csv("data/audio_ds.csv", index=False)

    # df = pd.read_csv("data/processed_audios_2.csv")
    
    # X = df.drop('label', axis=1)
    # y = df['label']
    
    # # Extract groups for the split (source files)
    # groups = X['source']
    
    # print(f"Number of unique source files: {len(groups.unique())}")
    # print(f"Class distribution: {sum(y)} / {len(y)} = {sum(y) / len(y):.4f}")
    
    # # Perform a group-based split to ensure sources appear in only one split
    # gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    
    # # Get indices for the train-test split
    # train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    # # Apply the split
    # X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # # Verify we have properly separated sources
    # train_sources = set(X_train['source'])
    # test_sources = set(X_test['source'])
    # overlap = train_sources.intersection(test_sources)
    # print(f"Sources in training set: {len(train_sources)}")
    # print(f"Sources in testing set: {len(test_sources)}")
    # print(f"Overlapping sources: {len(overlap)}")
    # print(f"Train class distribution: {sum(y_train)} / {len(y_train)} = {sum(y_train) / len(y_train):.4f}")
    # print(f"Test class distribution: {sum(y_test)} / {len(y_test)} = {sum(y_test) / len(y_test):.4f}")
    
    # X_train = X_train.drop('source', axis=1)
    # X_test = X_test.drop('source', axis=1)
    
    # cols = X_train.columns

    # lr_text = train_evaluate_model(
    #     X_train, X_test, 
    #     y_train, y_test,
    #     "Logistic Regression (Text Messages)",
    #     LogisticRegression(max_iter=10000, class_weight='balanced', random_state=42)
    # )

    # # Get indices sorted by absolute coefficient values in descending order
    # sorted_indices = np.argsort(-np.abs(lr_text.coef_[0]))  # Note the negative sign for descending order
    
    # # Get top 5 features and their coefficients
    # top5_features = [cols[i] for i in sorted_indices[:5]]
    # top5_coefficients = lr_text.coef_[0][sorted_indices[:5]]
    
    # print("\nTop 5 Features by Importance:")
    # for i, (feature, coef) in enumerate(zip(top5_features, top5_coefficients)):
    #     print(f"{i+1}. {feature}: {coef:.6f}")

    # rf_text = train_evaluate_model(
    #     X_train, X_test, 
    #     y_train, y_test,
    #     "Random Forest (Text Messages)",
    #     RandomForestClassifier(
    #         class_weight='balanced',
    #         random_state=42
    #     )
    # )


    # # Perform permutation importance test
    # print("\nPerforming permutation importance test for Random Forest model...")
    # perm_importance = permutation_importance(rf_text, X_test, y_test, 
    #                                      n_repeats=10)
    
    # # Sort features by importance
    # sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # # Display top 5 features
    # print("\nTop 5 Features by Permutation Importance:")
    # for i in range(min(5, len(sorted_idx))):
    #     feature_idx = sorted_idx[i]
    #     print(f"{i+1}. {X.columns[feature_idx]}: {perm_importance.importances_mean[feature_idx]:.6f} ± {perm_importance.importances_std[feature_idx]:.6f}")

    # hgbt_text = train_evaluate_model(
    #     X_train, X_test, 
    #     y_train, y_test,
    #     "HistGradientBoosting (Text Messages)",
    #     HistGradientBoostingClassifier(max_iter=100, random_state=42, class_weight='balanced')
    # )
    
    # # Perform permutation importance test
    # print("\nPerforming permutation importance test for HGBT model...")
    # perm_importance = permutation_importance(hgbt_text, X_test, y_test, 
    #                                      n_repeats=10)
    
    # # Sort features by importance
    # sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # # Display top 5 features
    # print("\nTop 5 Features by Permutation Importance:")
    # for i in range(min(5, len(sorted_idx))):
    #     feature_idx = sorted_idx[i]
    #     print(f"{i+1}. {X.columns[feature_idx]}: {perm_importance.importances_mean[feature_idx]:.6f} ± {perm_importance.importances_std[feature_idx]:.6f}")


    # df1 = pd.read_csv("data/test2_audios.csv")
    # X1 = df1.drop('label', axis=1)
    # y1 = df1['label']
    
    # print("\nTest 2")
    # evaluate_model(X1, y1, lr_text)
    # evaluate_model(X1, y1, rf_text)
    # evaluate_model(X1, y1, hgbt_text)

    # print("\nTest 3")
    # for source in test_sources:
    #     df_test = df[df['source'] == source]
    #     X_test = df_test.drop(['label', 'source'], axis=1)
    #     y_test = df_test['label']
        
    #     print("label = ", list(y_test)[0])
    #     evaluate_model(X_test, y_test, lr_text)
    #     evaluate_model(X_test, y_test, rf_text)
    #     evaluate_model(X_test, y_test, hgbt_text)
    #     print("===================================")
    