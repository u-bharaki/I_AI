import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_file_final.csv")
df["Diagnosis"] = df["Diagnosis"].str.upper()
gender_percent = df["Patient Sex"].value_counts(normalize=True) * 100
gender_labels = gender_percent.index.tolist()
gender_values = gender_percent.values.tolist()
disease_percent = df["Diagnosis"].value_counts(normalize=True) * 100
class_labels = disease_percent.index.tolist()
class_values = disease_percent.values.tolist()
ages = df["Patient Age"]

# GRAFİK 1: CİNSİYET
plt.figure(figsize=(6,6))
plt.pie(gender_values, labels=gender_labels, autopct='%1.1f%%', startangle=90)
plt.title("Gender Distribution (from CSV)")
plt.show()

# GRAFİK 2: HASTALIK
plt.figure(figsize=(10,5))
plt.bar(class_labels, class_values)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Percentage (%)")
plt.title("Disease Class Distribution (from CSV)")
plt.tight_layout()
plt.show()

# GRAFİK 3: YAŞ DAĞILIMI
plt.figure(figsize=(12,6))
plt.hist(ages, bins=20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution (from CSV)")
plt.grid(True)
plt.show()

# GRAFİK 4: HASTALIKLARA GÖRE YAŞ DAĞILIMI 
plt.figure(figsize=(14,6))
plt.scatter(df["Diagnosis"], df["Patient Age"], alpha=0.3)
plt.title("Age Scatter Plot per Disease Class (from CSV)")
plt.xlabel("Disease Class")
plt.ylabel("Age")
plt.grid(True)
plt.show()
