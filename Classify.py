import pandas as pd
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

def performance_test(model_labels, real_labels):
    TP = TN = FP = FN = 0
    for model_label, real_label in zip(model_labels, real_labels) :
        if model_label == real_label == 1 :
            TP += 1
        elif model_label == real_label == 0:
            TN += 1
        elif model_label == 1 and real_label == 0 :
            FP += 1
        elif model_label == 0 and real_label == 1 :
            FN += 1
    accuracy = (TP+TN) / (TP+TN+FP+FN)  # 0.8521675240646968      
    precision = TP/(TP+FP)  # 0.7734257276936505
    recall = TP/(TP+FN)   # 0.9961587223971186
    # check if 1-0 model label is inverse
    if accuracy < 0.5 :
        accuracy = 1 - accuracy
        precision = 1 - precision
        recall = 1 - recall
    F1 = 2* (precision*recall) / (precision+recall) # 0.8707748135205967
    print('Accuracy:', accuracy, '\nF1-Score:', F1, '\nPrecision:', precision, '\nRecall:', recall)
    
def read_dataset(dataset_url):
    df = pd.read_csv(dataset_url)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    label_copy = df['label'].copy()
    df.drop('label', axis=1, inplace=True)
    return df, label_copy

def resample_dataset(df, data_labels):
    sm = SMOTE(random_state=42)
    data_resampled, label_resampled = sm.fit_resample(df, data_labels)
    return data_resampled, label_resampled

def main():
    normal_dataset_url = 'kddcup_normal.csv'
    dataframe, labelsframe = read_dataset(normal_dataset_url)
    data_resampled, labels_resampled = resample_dataset(dataframe, labelsframe)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data_resampled)
    performance_test(kmeans.labels_, labels_resampled.values)

if __name__ == '__main__':
    main()
