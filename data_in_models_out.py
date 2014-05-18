import pdb
import string
from sgmllib import *
from sklearn.feature_extraction.text import *
from gensim import *
from gensim.models import *
from scipy.sparse import *
from sklearn import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import DecisionTreeClassifier as DecTree
from sklearn.ensemble import *
from sklearn.lda import *
from sklearn.neural_network import *
from sklearn.feature_selection import chi2, SelectKBest
import nltk.data, nltk.tag
from nltk import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import *
from nltk.stem.porter import *
from nltk.corpus import brown

# SGML parser for scraping initial document
class mySGMLParser(SGMLParser):
  # Initial custom SGMLParser
  def __init__(self):
    # Initialise parent
    SGMLParser.__init__(self)
    # Content will contain the 'body' of each tag
    self.content = []
    self.lastTag = ''

  def unknown_starttag(self, thisTag, thisAttributes):
    if (thisTag == 'd'):
        return
    self.lastTag = thisTag
    # Start a new dictionary for initial article
    if (thisTag == 'reuters'):
        thisAttributes.append(('tag', thisTag))
        self.content.append(dict(thisAttributes))

  # handle_data gets called for every line of the data
  def handle_data(self, data):
    # Ignore preamble lines and newlines
    if (len(self.content) == 0) or (data == '\n'):
        return
    # Store body data for this article
    if (self.lastTag == 'body' or self.lastTag == 'title'):
        try:
            self.content[-1]['body'] += ' ' + data
        except KeyError:
            self.content[-1]['body'] = data
    if (self.lastTag == 'topics'):
        try:
            self.content[-1]['topic'] += ' ' + data
        except KeyError:
            self.content[-1]['topic'] = data

def readFiles(parser):
    fileStr = 'reut2-0N.sgm'
    for i in range(22):
        thisFileStr = fileStr.replace('N', '0' + str(i) if (i < 10) else str(i))
        print thisFileStr
        curFile = open(thisFileStr, 'r')
        for line in curFile:
            parser.feed(line)

def articleIsValid(article):
    return {'body', 'topic'}.issubset(article.keys()) and len(article['topic'])> 0

def isTrainingArticle(article):
    return article['lewissplit'] == 'TRAIN'

def makeMatrix(x, numColumns):
    matrix = [0] * numColumns
    for i in x:
        matrix[i[0]] = i[1]
    return matrix

importantTopics = ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade', 'wheat']
def mostImportantArticle(article):
    overLap = set(importantTopics) & set(article['topic'])
    return len(overLap) != 0

def removeUnimportantTopics(article):
    overLap = set(importantTopics) & set(article['topic'])
    article['topic'] = list(overLap)

def splitTopics(article):
    try:
        article['topic'] = article['topic'].strip().split()
    except KeyError:
        article['topic'] = []

stemmer = PorterStemmer()
interestingTags = ['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'VB', 'VBD',
                    'VBG', 'VBN', 'VBP', 'VBZ']
tagger = UnigramTagger(brown.tagged_sents())

def tag_and_stem(body):
    currentArticle = tagger.tag(nltk.word_tokenize(body.lower()))
    trimmed = [x[0] for x in currentArticle if x[1] in interestingTags]
    body = stemmer.stem(' '.join(trimmed))
    return body

# -- MAIN SCRIPT BEGINS HERE

trainingArticles = []
trainingTopics = []
testArticles = []
testTopics = []

# -- READ INPUT USING SGMLPARSER
myParser = mySGMLParser()
readFiles(myParser)

# -- FILTERING AND PREPROCESSING STAGE
print "Gathering articles.."
for article in myParser.content:
    if articleIsValid(article):
        splitTopics(article)
        removeUnimportantTopics(article)
        if article['topic'] == []:
            continue
        if isTrainingArticle(article):
            #trainingArticles.append(tag_and_stem(article['body']))
            trainingArticles.append(article['body'])
            trainingTopics.append(article['topic'])
        else:
            #testArticles.append(tag_and_stem(article['body']))
            testArticles.append(article['body'])
            testTopics.append(article['topic'])

# -- DEFINING THE SET OF VECTORIZERS FOR THIS RUN
vectorizers = []
#vectorizers.append(TfidfVectorizer(stop_words='english',
#    decode_error=u'ignore'))
#vectorizers.append(TfidfVectorizer(stop_words='english',
#    decode_error=u'ignore', max_features=1000))
vectorizers.append(CountVectorizer(stop_words='english',
    decode_error=u'ignore'))
vectorizers.append(CountVectorizer(stop_words='english',
    decode_error=u'ignore', max_features=1000))

# -- DEFINING THE SET OF VECTORIZERS FOR THIS RUN
classifiers = []
#classifiers.append(RandomForestClassifier())
#classifiers.append(DecTree(max_depth=10))
classifiers.append(SVC(kernel='linear'))
#classifiers.append(LinearSVC())
#classifiers.append(KNeighborsClassifier(3))
#classifiers.append(OneVsRestClassifier(MultinomialNB()))
#classifiers.append(DBSCAN())
#classifiers.append(KMeans(n_clusters=2))
#classifiers.append(KMeans(n_clusters=4))
#classifiers.append(KMeans(n_clusters=6))
#classifiers.append(DBSCAN(min_samples=10))
#classifiers.append(Ward(n_clusters=8))

#ls = [100, 100]

# -- TRAINING AND TESTING CLASSIFIERS
thisTrainingArticles = trainingArticles
thisTestArticles = testArticles
vectorizerNumber = 0
# -- LOOP OVER VECTORIZERS
for vectorizer in vectorizers:
    print 'vectorizer: ', vectorizerNumber
    vectorizerNumber += 1
    classifierNumber = 0
    # -- LOOP OVER CLASSIFIERS
    for classifier in classifiers:
        print 'classifier: ', classifierNumber
        classifierNumber += 1
        # -- RUN VECTORIZER ON TRAINING DATA
        trainingVector = vectorizer.fit_transform(thisTrainingArticles)
        #ch2 = SelectKBest(chi2, ls[vectorizerNumber-1])
        #trainingVector = ch2.fit_transform(trainingVector, thisTrainingArticles)
        #trainingCorpus = matutils.Sparse2Corpus(trainingVector, documents_columns=False)
        #numTopics = ls[vectorizerNumber-1]
        #model = ldamodel.LdaModel(trainingCorpus, num_topics=numTopics)
        #modelResults = [model[m] for m in trainingCorpus]
        #modelMatrix = [makeMatrix(m, numTopics) for m in modelResults]
        #sparseMatrix = coo_matrix(modelMatrix)
        #modelMatrixStack = hstack([trainingVector, sparseMatrix])

        # -- RUN VECTORIZER ON TEST DATA
        #testVector = vectorizer.transform(thisTestArticles)
        #testVector = ch2.transform(testVector)
        testVector = vectorizer.transform(thisTestArticles)
        #testVectorCorpus = matutils.Sparse2Corpus(testVector, documents_columns=False)
        #testModelResults = [model[m] for m in testVectorCorpus]
        #testMatrix = [makeMatrix(m, numTopics) for m in testModelResults]
        #testSparseMatrix = coo_matrix(testMatrix)
        #testMatrixStack = hstack([testVector, testSparseMatrix])


        # -- FIT CLASSIFIER TO TRAINING DATA AND TEST ON TEST DATA
        #classifier.fit(sparseMatrix.toarray(), trainingTopics)
        #classifiedTestData = classifier.predict(testMatrix)
        #classifier.fit(modelMatrixStack.toarray(), trainingTopics)
        #classifier.fit(modelMatrixStack.toarray())
        #classifiedTestData = classifier.predict(testMatrixStack.toarray())
        #classifier.fit(trainingVector.toarray(), trainingTopics)
        classifier.fit(trainingVector, trainingTopics)
        classifiedTestData = classifier.predict(testVector)
        print accuracy_score(testTopics, classifiedTestData)
        print recall_score(testTopics, classifiedTestData)
        print recall_score(testTopics, classifiedTestData, average='micro')
        print recall_score(testTopics, classifiedTestData, average='macro')
        print precision_score(testTopics, classifiedTestData)
        print precision_score(testTopics, classifiedTestData, average='micro')
        print precision_score(testTopics, classifiedTestData, average='macro')
        print f1_score(testTopics, classifiedTestData)
        print f1_score(testTopics, classifiedTestData, average='micro')
        print f1_score(testTopics, classifiedTestData, average='macro')
        #classifiedTestData = classifier.labels_
        #print adjusted_rand_score(classifiedTestData, trainingTopics)
        #print adjusted_mutual_info_score(classifiedTestData, trainingTopics)
        #print homogeneity_score(classifiedTestData, trainingTopics)
        #print completeness_score(classifiedTestData, trainingTopics)
        #print v_measure_score(classifiedTestData, trainingTopics)
        #try:
        #    #print silhouette_score(modelMatrixStack.toarray(), classifiedTestData,
        #    #    metric='euclidean')
        #    print silhouette_score(trainingVector, classifiedTestData,
        #        metric='euclidean')
        #except ValueError:
        #    print 'error'
        #print len(set(classifiedTestData)) - (1 if -1 in classifiedTestData else 0)

