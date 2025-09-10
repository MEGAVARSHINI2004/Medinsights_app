import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/HAM10000_metadata.csv")

# Disease distribution
sns.countplot(x="dx", data=df, palette="Set2")
plt.xticks(rotation=45)
plt.title("Disease Distribution")
plt.show()

# Age distribution
sns.histplot(df["age"].dropna(), bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Gender balance
sns.countplot(x="sex", data=df, palette="coolwarm")
plt.title("Gender Distribution")
plt.show()
