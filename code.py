# Problem: Detecting fake news with a PassiveAggressive Classifier and TfidfVectorizer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



fake_df = pd.read_csv('E:\VS Code\Fake News\dataset/Fake.csv')
true_df = pd.read_csv('E:\VS Code\Fake News\dataset/True.csv')


fake_df['label'] = 0 
true_df['label'] = 1 
df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)


X = df['text'] 
y = df['label']


tf_idf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7) 
X_tf_idf = tf_idf_vectorizer.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X_tf_idf, y, test_size=0.2, random_state=42)


model = PassiveAggressiveClassifier(max_iter=10)
model.fit(X_train, y_train)


predictions = model.predict(X_val)
accuracy_score = accuracy_score(y_val, predictions)
print("Accuracy score:\n", accuracy_score)
confusion_matrix = confusion_matrix(y_val, predictions)
print("Confusion Matrix:\n", confusion_matrix)
classification_report = classification_report(y_val, predictions)
print("Classfication Report:\n", classification_report)



plt.figure(figsize=(7,5))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Confusion Matrix')
plt.show()
