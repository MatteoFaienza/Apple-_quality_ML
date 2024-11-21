
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

# 1 Leggo il csv

file_path = "apple_quality_last.csv" 

df_apple_quality = pd.read_csv(file_path)

print(df_apple_quality.isna().sum().sum())
print(df_apple_quality.info())
print(df_apple_quality.describe())

# 2 Controllo correttezza del dataset
# 2.1 Elimimo dati nulli

#df_apple_quality = df_apple_quality.interpolate()
#df_apple_quality.dropna(subset=['Acidity', "Quality","Crunchiness"], inplace=True)

df_apple_quality = df_apple_quality.dropna()

# 2.2 Elimino dati duplicati

df_apple_quality.drop_duplicates()

print(df_apple_quality.info())

# 2.3 Controllo coerenza stringhe

#for i in range(len(acidity_columns)) :
    #if not isinstance(acidity_columns[i], float):
        #print(acidity_columns)

df_apple_quality["Acidity"] = df_apple_quality["Acidity"].astype(float) 

# Vedo i diversi elemnti in una colonna
df_apple_quality["Quality"].unique()

df_apple_quality["Quality"] = df_apple_quality["Quality"].str.strip("_")
df_apple_quality["Quality"] = df_apple_quality["Quality"].str.lower()

# 3 Analisi dati

#Elimino la colonna dell Id che non serve
df_apple_quality = df_apple_quality.drop("A_id", axis = 1)


for column_name in df_apple_quality.columns[0:-1]:

    mean = df_apple_quality[column_name].mean()
    median = df_apple_quality[column_name].median()
    variance =df_apple_quality[column_name].std() 

    print("--",column_name)
    print(f" Media    : {mean}")
    print(f" Mediana  : {median}")
    print(f" Varianza : {variance}")

# 4 Preprocessing

normalizer = MinMaxScaler()
label_encoder = LabelEncoder()

for column_feut in df_apple_quality :
    
    if df_apple_quality[column_feut].dtype in ['int64', 'float64']:
        df_apple_quality[column_feut] = normalizer.fit_transform(df_apple_quality[[column_feut]])

    elif df_apple_quality[column_feut].dtype == 'object':
        df_apple_quality[column_feut] = label_encoder.fit_transform(df_apple_quality[column_feut])

# #print(df_apple_quality)

# Definizione di una colormap personalizzata
corr_map_color = LinearSegmentedColormap.from_list("corr_map_color", ["green", "white", "red"])

# Calcolo della matrice di correlazione
correlation_matrix = df_apple_quality.corr()

# grafico con seaborn
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap=corr_map_color)
plt.title('Correlation heatmap')
plt.show()

# 5 Preparazione dei dati

X = df_apple_quality[["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]]
y = df_apple_quality["Quality"]

# 5.1 Split dei dati
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28, test_size = 0.2)

# 6 Scelgo modello machine learning
svm = SVC(kernel='rbf')

# 6.1 Addestramento
svm.fit(X_train, y_train)

# 6.2 Predizioni
y_prediction = svm.predict(X_test)

# 7 Valutazione modello
accuracy = accuracy_score(y_test, y_prediction)
report_class = classification_report(y_test, y_prediction)
print("Accuracy ", accuracy )
print(report_class)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_prediction)
print(conf_matrix)

plt.figure(figsize =(10,8))
sns.heatmap(conf_matrix, cmap='Greens', fmt="d", annot=True)
plt.show()

