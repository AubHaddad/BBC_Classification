from os import listdir
from os.path import isdir, isfile, join
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import operator
from pprint import pprint
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
#Classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score, classification_report, average_precision_score, balanced_accuracy_score

path = "G:\\PedaProjects\\NLP\\BBCProject\\bbc"
listDossiers  = [directory for directory in listdir(path) if isdir(join(path, directory))]
listDossiers.sort()
listDossiersPath = [join(path,directPath) for directPath in listDossiers ]
def getArticles(doss):
	listArticlesNames = [article for article in listdir(doss) if isfile(join(doss,article))];
	listArticlesCountent = []
	for article in listArticlesNames:
		if article.endswith(".txt"):
			with open(join(doss,article),"r") as fd:
				listArticlesCountent.append(fd.read())
	return listArticlesCountent
def getArticlesByClasse(classe):
	classe.lower()
	print("\nRécupération des articles :",classe)
	index = -1
	l_articles =[]
	if classe in listDossiers :
		index = listDossiers.index(classe)
		if index != -1 :
			l_articles = getArticles(listDossiersPath[index])
	return l_articles

def preprocess(articles):
	print("\nPreprocessing started .....")
	list_articles_words=[]
	for article in articles:
		list_words=[]
		article=article.lower()
		##
		#print("######## Article before : ", article)
		article=re.sub(r"[^a-z ]*","",article)
		article=re.sub(r"[\s]+"," ",article)
		# print("!!!!!!!!! Article apres : ", article)
		article_words=word_tokenize(article)
		for word in article_words:
			if(word not in stop_words):
				word=stemmer.stem(word)
				#word = stemmer.lemmatize(word)
				list_words.append(word)
		#print("tweet words",list_words)
		list_articles_words.append(" ".join(list_words))
	print("\nPreprocessing finished.")
	return list_articles_words


print("\nInitializing......\n")
businessArticles = getArticlesByClasse("business")
entertainmentArticles = getArticlesByClasse("entertainment")
politicsArticles = getArticlesByClasse("politics")
sportArticles = getArticlesByClasse("sport")
techArticles = getArticlesByClasse("tech")

allart = businessArticles+entertainmentArticles+politicsArticles+sportArticles+techArticles

stop_words=set(stopwords.words("english"))
stemmer=PorterStemmer()
#stemmer = WordNetLemmatizer()


allart = preprocess(allart)

"""
countvectorizer=CountVectorizer()
countvector=countvectorizer.fit(allart)
countvector=countvectorizer.transform(allart)
X=countvector
"""
tfidfvectorizer=TfidfVectorizer()
tfidfvector=tfidfvectorizer.fit(allart)
tfidfvector=tfidfvectorizer.transform(allart)
X = tfidfvector


Y = ["business" for i in range(len(businessArticles))] \
    + ["entertainment" for i in range(len(entertainmentArticles))] \
    + ["politics" for i in range(len(politicsArticles))] \
    + ["sport" for i in range(len(sportArticles))] \
    + ["tech" for i in range(len(techArticles))]
labelEncoder = LabelEncoder()
Y = labelEncoder.fit(Y).transform(Y)


print("nombre Business Articles : ",len(businessArticles))
print("nombre entertainment Articles : ",len(entertainmentArticles))
print("nombre politics Articles : ",len(politicsArticles))
print("nombre sport Articles : ",len(sportArticles))
print("nombre tech Articles : ",len(techArticles))

print("\nlength of the longest article  :",len(max(allart, key=len).split()))
print("\nlength of the shortest article :",len(min(allart, key=len).split()))


print("\nCreating train and test Data .....")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100, shuffle="true")

accuracy_dict = dict()
f1_micro_dict = dict()
f1_macro_dict = dict()
print("\n-!-!-!-!-!-!-!-!-!-!-!-! MultinomialNB !-!-!-!-!-!-!-!-!-!-!-!-")
clfMultinomialNB = MultinomialNB()
print("\nTraining .....")
modelMultinomialNB = clfMultinomialNB.fit(X_train,Y_train)
print("\nTraining finished.")
Y_predict = modelMultinomialNB.predict(X_test)
print("\n######################################################\n")
accuracy_dict['MultinomialNB'] = round(balanced_accuracy_score(Y_test,Y_predict),4)*100
f1_micro_dict['MultinomialNB'] = f1_score(Y_test,Y_predict, average='micro')
f1_macro_dict['MultinomialNB'] = f1_score(Y_test,Y_predict, average='macro')

matrix = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)
print("confusion matrix \n", matrix)
print(report)


print("\n-!-!-!-!-!-!-!-!-!-!-!-! BernoulliNB !-!-!-!-!-!-!-!-!-!-!-!-")

clfBernoulliNB = BernoulliNB()
print("\nTraining .....")
modelBernoulliNB = clfBernoulliNB.fit(X_train,Y_train)
print("\nTraining finished.")
Y_predict = modelBernoulliNB.predict(X_test)
print("\n######################################################\n")

accuracy_dict['BernoulliNB'] = round(balanced_accuracy_score(Y_test,Y_predict),4)*100
f1_micro_dict['BernoulliNB'] = f1_score(Y_test,Y_predict, average='micro')
f1_macro_dict['BernoulliNB'] = f1_score(Y_test,Y_predict, average='macro')

matrix = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)
print("confusion matrix \n", matrix)
print(report)

print("\n-!-!-!-!-!-!-!-!-!-!-!-! DecisionTreeClassifier !-!-!-!-!-!-!-!-!-!-!-!-")


clfDecisionTreeClassifier = DecisionTreeClassifier()
print("\nTraining .....")
modelDecisionTreeClassifier = clfDecisionTreeClassifier.fit(X_train,Y_train)
print("\nTraining finished.")
Y_predict = modelDecisionTreeClassifier.predict(X_test)
print("\n######################################################\n")

accuracy_dict['DecisionTreeClassifier'] = round(balanced_accuracy_score(Y_test,Y_predict),4)*100
f1_micro_dict['DecisionTreeClassifier'] = f1_score(Y_test,Y_predict, average='micro')
f1_macro_dict['DecisionTreeClassifier'] = f1_score(Y_test,Y_predict, average='macro')

matrix = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)
print("confusion matrix \n", matrix)
print(report)

print("\n-!-!-!-!-!-!-!-!-!-!-!-! KNeighborsClassifier !-!-!-!-!-!-!-!-!-!-!-!-")

clfKNeighborsClassifier = KNeighborsClassifier()
print("\nTraining .....")
modelKNeighborsClassifier = clfKNeighborsClassifier.fit(X_train,Y_train)
print("\nTraining finished.")
Y_predict = modelKNeighborsClassifier.predict(X_test)
print("\n######################################################\n")

accuracy_dict['KNeighborsClassifier'] = round(balanced_accuracy_score(Y_test,Y_predict),4)*100
f1_micro_dict['KNeighborsClassifier'] = f1_score(Y_test,Y_predict, average='micro')
f1_macro_dict['KNeighborsClassifier'] = f1_score(Y_test,Y_predict, average='macro')

matrix = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)
print("confusion matrix \n", matrix)
print(report)

print("\n-!-!-!-!-!-!-!-!-!-!-!-! LinearSVC !-!-!-!-!-!-!-!-!-!-!-!-")

clfLinearSVC = LinearSVC(multi_class="crammer_singer")
print("\nTraining .....")
modelLinearSVC = clfLinearSVC.fit(X_train,Y_train)
print("\nTraining finished.")
Y_predict = modelLinearSVC.predict(X_test)
print("\n######################################################\n")

accuracy_dict['LinearSVC'] = round(balanced_accuracy_score(Y_test,Y_predict),4)*100
f1_micro_dict['LinearSVC'] = f1_score(Y_test,Y_predict, average='micro')
f1_macro_dict['LinearSVC'] = f1_score(Y_test,Y_predict, average='macro')

matrix = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)
print("confusion matrix \n", matrix)
print(report)

print("\n-!-!-!-!-!-!-!-!-!-!-!-! RandomForestClassifier !-!-!-!-!-!-!-!-!-!-!-!-")

clfRandomForestClassifier = RandomForestClassifier()
print("\nTraining .....")
modelRandomForestClassifier = clfRandomForestClassifier.fit(X_train,Y_train)
print("\nTraining finished.")
Y_predict = modelRandomForestClassifier.predict(X_test)
print("\n######################################################\n")

accuracy_dict['RandomForestClassifier'] = round(balanced_accuracy_score(Y_test,Y_predict),4)*100
f1_micro_dict['RandomForestClassifier'] = f1_score(Y_test,Y_predict, average='micro')
f1_macro_dict['RandomForestClassifier'] = f1_score(Y_test,Y_predict, average='macro')

matrix = confusion_matrix(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)
print("confusion matrix \n", matrix)
print(report)


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

accuracy_sorted =sorted(accuracy_dict.items(), key=operator.itemgetter(1), reverse=True)
f1_macro_dict_sorted =sorted(f1_macro_dict.items(), key=operator.itemgetter(1), reverse=True)
f1_micro_dict_sorted =sorted(f1_micro_dict.items(), key=operator.itemgetter(1), reverse=True)
print("\n-----Accuracy : ")
for acc in accuracy_sorted:
	print(acc[0]," : ", round(float(acc[1]),4), "%")

print("\n-----F1 score macro : ")
for f1mac in f1_macro_dict_sorted:
	print(f1mac[0]," : ", round(float(f1mac[1]*100),4), "%")
print("\n-----F1 score micro : ")
for f1mic in f1_micro_dict_sorted:
	print(f1mic[0]," : ", round(float(f1mic[1]*100),4), "%")
