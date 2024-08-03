import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

#Phase-1
#This dictionary maps the names of different data sources (yelp, amazon, imdb) to their respective file paths
filepath_dict = {'yelp': 'data/sentiment_analysis/yelp_labelled.txt',
                'amazon' : 'data/sentiment_analysis/amazon_cells_labelled.txt',
                'imdb': 'data/sentiment_analysis/imdb_labelled.txt'}

#Data frame created from each file
df_list = []

#Loop through each key-value pair
for source, filepath in filepath_dict.items():
    #Read the file at the given file path into DataFrame using pd.read_csv().
    #The names = ['sentence','label'] argument specifices the column names, and sep='\t' indicates that the file is tab-separated.
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')

    #Add a new column
    df['source'] = source #add another column filled with the source name
    #Append the Data Frame
    df_list.append(df)

#Concatenate all the DataFrames in df_list into a single DataFrame df.
df = pd.concat(df_list)

#This prints the first row of the DataFrame df using iloc[0] which accesses the first row by its index.
#print(df.iloc[0])

#print all rows
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)
#print(df)

#Phase-2
#DataFrame Creation
#Filtering For Yelp Data
#Extracting Setences and Labels
#Splitting the Data

#Filter the DataFrame for Yelp data
df_yelp = df[df['source'] == 'yelp']

#Extract sentences and labels
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

#Split the data into training and testing sets
sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

#print the first few training setences and labels
print("Training setences:", sentences_train[:5])
print("Training labels:", y_train[:5])

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
X_train
print(X_train)
