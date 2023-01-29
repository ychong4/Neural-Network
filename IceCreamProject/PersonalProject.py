import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

df =pd.read_csv("reviews.csv")
product =pd.read_csv("products.csv")
df.fillna(0)
print(df.head())

np.random.seed(1000)

#Brand histogram plot
hist = px.histogram(df, x="brand", template="simple_white")
hist.update_layout(title_text = 'Ice cream brand')
hist.write_image('histogram_brand.png')
#hist.show()


#Rating histogram plot
hist = px.histogram(df, x="stars", template="simple_white")
hist.update_layout(title_text = 'Review Stars')
hist.write_image('histogram_review.png')
#hist.show()

# Create stopword list:
#stopwords = set(STOPWORDS)
#stopwords.update(["br", "href"])

wordnet = WordNetLemmatizer()

# WordCloud for reviews
stopwords = set(STOPWORDS)
textt = " ".join(word for word in df.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud.png')
#plt.show()

# WordCloud for titles
stopwords = set(STOPWORDS)
textt = " ".join(str(word) for word in df.title)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_title.png')
#plt.show()

# WordCloud for ingredients
stopwords = set(STOPWORDS)
textt = " ".join(word for word in product.ingredients)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_ingredients.png')
#plt.show()

# WordCloud for flavors
stopwords = set(STOPWORDS)
flavor = " ".join(str(word) for word in product.name)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(flavor)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_flavors.png')
#plt.show()



# Data Transformation
# Assign sentiment based on ratings
# Positive sentiment 1 = [5] ratings
# Neutral sentiment 0 = [3,4] ratings
# Negative sentiment -1 = [1,2] ratings

def sentiment(row):
    if row['stars'] == 5:
        return 1
    if row['stars'] < 3:
        return -1
    else:
        return 0

df['sentiment'] = df.apply(lambda row: sentiment(row), axis = 1)

bj = df.loc[df['brand']=='bj']
breyers = df.loc[df['brand']=='breyers']
hd = df.loc[df['brand']=='hd']
talenti = df.loc[df['brand']=='talenti']


# 1) Examine the relationship between season and the number of ratings of customer -- rating count distribute among months (bar chart)

categoryarray = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
hist = px.histogram(df, x="Month", template="simple_white", category_orders={"Month": categoryarray})
hist.update_layout(title_text = 'Rating counts distribution among months')
hist.write_image('1.png')
#hist.show()

# We observe higher rating counts during summer season and lower rating counts during winter season. This makes intuitive sense
# as there are more customers during the summer season.

# 2) Examine the relationship between custome rating and helpfulness count -- helpfulness count distribute among rating 1 to 5 (bar charts)

fig = px.bar(df, x="stars", y="helpful_yes", template="simple_white")
fig.update_layout(title_text = 'helpful_yes counts distribution among ratings')
fig.write_image('2_helpful_yes.png')
#fig.show()

fig = px.bar(df, x="stars", y="helpful_no", template="simple_white")
fig.update_layout(title_text = 'helpful_no counts distribution among ratings')
fig.write_image('2_helpful_no.png')
#fig.show()

fig = px.box(df, x="stars", y="helpful_no", template="simple_white")
fig.update_layout(title_text = 'helpful_no boxplots among ratings')
fig.write_image('2_helpful_no_boxplot.png')
#fig.show()

fig = px.box(df, x="stars", y="helpful_yes", template="simple_white")
fig.update_layout(title_text = 'helpful_yes boxplots among ratings')
fig.write_image('2_helpful_yes_boxplot.png')
#fig.show()

# For helpful_yes count, rating 5 has highest total counts. For helpful_no count, rating 1 has highest total counts. However, we observe more "bigger pieces"
# in rating 1 for both help_yes count and help_no count. Then, we read the boxplots and find out that the average for both helpful_yes and helpful_no counts
# are higher in rating 1 and rating 2.


# 3) Examine the relationship between ice cream brand and customer ratings -- Customer ratings distribution among ice cream brand (Gar chart)

fig = px.bar(df, x="brand",  facet_col="stars", color = "brand", template="simple_white", category_orders={"stars": [1,2,3,4,5]})
fig.update_layout(title_text = 'Customer ratings distribution among ice cream brand')
fig.write_image('3.png')
#fig.show()

# From the plot, we observe that the customer ratings groups are distributed quite similarly among the ice cream brand. However, one noticable
# result is that the breyers ice cream brand has more '1' customer ratings.


# 4) Examine the relationship between review ratings and the review text (wordclouds)

positive = df[df['sentiment'] == 1]
neutral = df[df['sentiment'] == 0]
negative = df[df['sentiment'] == -1]


stopwords = set(STOPWORDS)
textt = " ".join(word for word in positive.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_text.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in neutral.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_text.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in negative.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_text.png')
#plt.show()

# 5) Examine the relationship between review ratings and the review title. (wordclouds)



stopwords = set(STOPWORDS)
titles = " ".join(str(word) for word in positive.title)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(titles)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_title.png')
#plt.show()


stopwords = set(STOPWORDS)
titles = " ".join(str(word) for word in neutral.title)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(titles)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_title.png')
#plt.show()


stopwords = set(STOPWORDS)
titles = " ".join(str(word) for word in negative.title)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(titles)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_title.png')
#plt.show()

# 6) Examine the relationship between review ratings and the ingredients. (wordclouds)
def sentiment_product(row):
    if  0 < row['rating'] <= 2.0:
        return 1
    if 2.0 < row['rating'] <= 3.0:
        return 0
    if 3.0 < row['rating'] <= 5.0:
        return -1

product['sentiment'] = product.apply(lambda row: sentiment_product(row), axis = 1)

positive_product = product[product['sentiment'] == 1]
neutral_product = product[product['sentiment'] == 0]
negative_product = product[product['sentiment'] == -1]

stopwords = set(STOPWORDS)
ingredients = " ".join(str(word) for word in positive_product.ingredients)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(ingredients)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_ingredients.png')
#plt.show()


stopwords = set(STOPWORDS)
ingredients = " ".join(str(word) for word in neutral_product.ingredients)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(ingredients)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_ingredients.png')
#plt.show()


stopwords = set(STOPWORDS)
ingredients = " ".join(str(word) for word in negative_product.ingredients)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(ingredients)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_ingredients.png')
#plt.show()

# 7) Examine the relationship between review ratings and the ice cream flavors. (wordclouds)

stopwords = set(STOPWORDS)
flavor = " ".join(str(word) for word in positive_product.name)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(flavor)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_flavors.png')
#plt.show()


stopwords = set(STOPWORDS)
flavor = " ".join(str(word) for word in neutral_product.name)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(flavor)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_flavors.png')
#plt.show()


stopwords = set(STOPWORDS)
flavor = " ".join(str(word) for word in negative_product.name)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(flavor)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_flavors.png')
#plt.show()


# 8) Build a data model to predict the review sentiment based on the review text.

def remove_punctuation(review):
    cleaned = "".join(x for x in review if x not in ("?", ".", ";", ":", "!", '"', "+", "-"))
    return cleaned

df['text'] = df['text'].apply(remove_punctuation)

dfNew = df[['text', 'sentiment']]
#print(dfNew.head())

# Split data to 75% training set and 25% test set
Train_set = dfNew.iloc[:16256, :]
Test_set = dfNew.iloc[16257:, :]

# count vectorizer:

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
train_matrix = vectorizer.fit_transform(Train_set['text'])
test_matrix = vectorizer.transform(Test_set['text'])

clf = RandomForestClassifier()
x_train = train_matrix
x_test = test_matrix
y_train = Train_set['sentiment']
y_test = Test_set['sentiment']

clf.fit(x_train,y_train)
pred = clf.predict(x_test)

# Measure the performance of the classifier using Accuracy, Precision, and Recall rate
accuracy = accuracy_score(pred, y_test)
precision = precision_score(pred, y_test, average='weighted')
cm = confusion_matrix(pred, y_test)
print(cm)

print("The model to predict the review sentiment based on the review text has accuracy rate: " +str(round(accuracy,3)))
print("The model to predict the review sentiment based on the review text has precision rate: " +str(round(precision,3)))


# 9) Examine the relationship between sentiment of the reviews and the help_yes_count and help_no count. (classifier)

# Data visualization
dfNew9 = df[['helpful_yes', 'helpful_no', 'sentiment']]
print(dfNew9)

fig = px.scatter(dfNew9, x="sentiment", y="helpful_yes", template="simple_white")
fig.update_layout(title_text = 'helpful_yes counts distribution among sentiment')
fig.write_image('9_helpful_yes_sentiment.png')
#fig.show()

fig = px.scatter(dfNew9, x="sentiment", y="helpful_no", template="simple_white")
fig.update_layout(title_text = 'helpful_no counts distribution among sentiment')
fig.write_image('9_helpful_no_sentiment.png')
#fig.show()


# Splitting data into 75% training set and 25% test set
helpful_yes_train_set = df.iloc[:16256, 8:9]
helpful_yes_test_set = df.iloc[16257:, 8:9]
helpful_no_train_set = df.iloc[:16256, 9:10]
helpful_no_test_set = df.iloc[16257:, 9:10]
sentiment_train_set = df.iloc[:16256, -1]
sentiment_test_set = df.iloc[16257:, -1]

knn_yes = KNeighborsClassifier(n_neighbors=10, metric = 'euclidean')

# Training the helpful_yes model
knn_yes.fit(helpful_yes_train_set,sentiment_train_set)
y_pred_yes = knn_yes.predict(helpful_yes_test_set)


# Measure the performance of the classifier using Accuracy, Precision, and Recall rate
accuracy = accuracy_score(y_pred_yes, sentiment_test_set)
precision = precision_score(y_pred_yes, sentiment_test_set, average='weighted')
cm = confusion_matrix(y_pred_yes, sentiment_test_set)
print(cm)

print("The helpful_yes model accuracy rate is: " +str(round(accuracy,3)))
print("The helpful_yes model precision rate is: " +str(round(precision,3)))



# Training the helpful_no model
knn_no = KNeighborsClassifier(n_neighbors=5, metric = 'euclidean')
knn_no.fit(helpful_no_train_set,sentiment_train_set)
y_pred_no = knn_no.predict(helpful_no_test_set)
print(y_pred_no)


# Measure the performance of the classifier using Accuracy, Precision, and Recall rate
accuracy = accuracy_score(y_pred_no, sentiment_test_set)
precision = precision_score(y_pred_no, sentiment_test_set, average='weighted')
cm = confusion_matrix(y_pred_no, sentiment_test_set)
print(cm)

print("The helpful_no model accuracy rate is: " +str(round(accuracy,3)))
print("The helpful_no model precision rate is: " +str(round(precision,3)))



#10) Analyzing sentiment of different ice cream companies from the reviews

positive_bj = bj[bj['sentiment'] == 1]
neutral_bj = bj[bj['sentiment'] == 0]
negative_bj = bj[bj['sentiment'] == -1]

positive_breyers = breyers[breyers['sentiment'] == 1]
neutral_breyers = breyers[breyers['sentiment'] == 0]
negative_breyers = breyers[breyers['sentiment'] == -1]

positive_talenti = talenti[talenti['sentiment'] == 1]
neutral_talenti = talenti[talenti['sentiment'] == 0]
negative_talenti = talenti[talenti['sentiment'] == -1]

positive_hd = hd[hd['sentiment'] == 1]
neutral_hd = hd[hd['sentiment'] == 0]
negative_hd = hd[hd['sentiment'] == -1]

stopwords = set(STOPWORDS)
textt = " ".join(word for word in positive_bj.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_bj.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in positive_breyers.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_breyers.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in positive_talenti.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_talenti.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in positive_hd.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_positive_hd.png')
#plt.show()


stopwords = set(STOPWORDS)
textt = " ".join(word for word in neutral_bj.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_bj.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in neutral_breyers.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_breyers.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in neutral_talenti.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_talenti.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in neutral_hd.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_neutral_hd.png')
#plt.show()


stopwords = set(STOPWORDS)
textt = " ".join(word for word in negative_bj.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_bj.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in negative_breyers.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_breyers.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in negative_talenti.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_talenti.png')
#plt.show()

stopwords = set(STOPWORDS)
textt = " ".join(word for word in negative_hd.text)
wordcloud = WordCloud(stopwords=stopwords, background_color = "white").generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud_negative_hd.png')
#plt.show()