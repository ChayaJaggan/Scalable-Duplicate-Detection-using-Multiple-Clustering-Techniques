import json
import re
import collections
import numpy as np
import random
import pandas as pd
import math
from itertools import combinations
from strsimpy.qgram import QGram
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from Levenshtein import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN





# Import data
print("Start")
with open('C:/Users/chaya/PycharmProjects/pythonProject/Computer Science/TVs-all-merged.json', 'r') as f:
    data = f.read()
    # print(data)
    for line in f:
        content = line.strip().split()
        print(content)

dict = json.loads(data)

# obtains a list of all 1624 products
# starts with key (even index), then value follows (uneven index)

dictList = []
for key, value in dict.items():
    for value in dict[key]:
        dictList.append(key)
        dictList.append(value)

# list of only the product descriptions
productDescriptions = dictList[1::2]


def bootstraps(data):
    #random.seed(1)
    seed = random.randint(0,1000)
    dictionarytrain = {}
    dictionarytest = {}
    unique = 0
    data = np.array(data)
    sampleSize = data.shape[0]
    index = [i for i in range(sampleSize)]
    indices = np.random.choice(index, replace=True, size=sampleSize)
    indices = list(set(indices))
    train = data[indices,]
    print('The number of train samples ' + str(len(train)))
    oob = list(set(index) - set(indices))
    test = np.array([])
    if oob:
        test = data[oob,]
    print('The number of test samples ' + str(len(test)))
    dictionarytrain['train'] = {'train': list(train)}
    dictionarytest['test'] = {'test': list(test)}
    return (dictionarytrain, dictionarytest)


train, test = bootstraps(productDescriptions)

train_items = []
for value in train['train'].values():
    for item in value:
        train_items.append(item)

test_items = []
for value in test['test'].values():
    for item in value:
        test_items.append(item)

# obtain title of products
def getTitle(productDescription):
    titles = []
    for product in productDescription:
        value = product['title']
        titles.append(value)
    return titles


# Process the titles
def preProcess(titles):
    processedTitle = []
    processedTitle2 = []
    processedTitle3 = []
    counter = {}
    for title in titles:
        # remove round brackets, backslash, and -
        words = ['/', "(", ")", " -", "Newegg.com", "TheNerds.net", "Best Buy", "tv", "TV", "Class", ";", "$"]
        for word in words:
            if word in title:
                title = title.replace(word, "")
        # change uppercase letters to lowercase
        for word in title:
            if type(word) == str:
                title = title.replace(word, word.lower())
        # normalize to inch
        typeInch = [' inch', 'inches', '-inch', '"', 'inch']
        for word in typeInch:
            if word in title:
                title = title.replace(word, "inch")
        # normalize to hz
        typeHz = [' hz', 'hertz', '-hz']
        for word in typeHz:
            if word in title:
                title = title.replace(word, "hz")
        # remove all variants of diagonal
        typeDiagonal = ['diag.', ' diag.' ' diagonally', ' diagonal widescreen', ' diagonal size', \
                        ' diag', 'diag', 'diagonal', ' diagonal', 'diagonal size']
        for word in typeDiagonal:
            if word in title:
                title = title.replace(word, "")
        if 'led-lcd' in title:
            title = title.replace('led-lcd', "ledlcd")
        processedTitle.append(title)
        for word in title.split():
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1
    # do not use the words that only appear once
    counter_one = {key: value for key, value in counter.items() if value == 1}

    for title in processedTitle:
        for word in counter_one.keys():
            if word in title.split():
                title = title.replace(word, '')
        processedTitle2.append(title)
    for title in processedTitle2:
        if title.startswith(' '):
            title = title[1:]
        if title.endswith(' '):
            title = title[:-1]
        processedTitle3.append(title)

    return processedTitle3



# dictionary of products as keys and a list as values
def getDictionary(list):
    keys = []
    for i in range(0, len(list)):
        keys.append(i)
    dictionary = {keys[i]: list[i] for i in range(len(keys))}
    return dictionary




# Returns the key-value pairs
def getKeyValue(productDescription):
    keyvalue = {}
    for i, product in enumerate(productDescription):
        keyvalue[i] = product['featuresMap']
    return keyvalue




# Processes a string, used to process the key-value pairs
def preprocess(string):
    # remove round brackets, backslash, and -
    # words = ['/', "(", ")", " -", " "]
    words = ['/', "(", ")", " -"]
    for word in words:
        if word in string:
            string = string.replace(word, "")
    # change uppercase letters to lowercase
    for word in string:
        if type(word) == str:
            string = string.replace(word, word.lower())
    # normalize inch
    typeInch = [' inch', 'inches', '-inch', '"', 'inch']
    for word in typeInch:
        if word in string:
            string = string.replace(word, "inch")
    # normalize to hz
    typeHz = [' hz', 'hertz', '-hz']
    for word in typeHz:
        if word in string:
            string = string.replace(word, "hz")
    typeWeight = [' lbs', 'pounds', ' pounds', 'lb', ' lb', 'lbs.', 'pound']
    for word in typeWeight:
        if word in string:
            string = string.replace(word, "lbs")
    return string

# Returns the processed values from the key-value pairs
def processKV(data_items):
    values = collections.defaultdict(list)
    keyValues = getKeyValue(data_items)
    for product, pairs in keyValues.items():
        for key, value in pairs.items():
            value = preprocess(value)
            if key in values:
                values[product] = value
            else:
                values[product] = value

    return values



# Returns the binary vector
def getBinaryVector(dictionary,items):
    modelWords_title = []
    modelWords_value = []
    keyvalue = getKeyValue(items)
    process = processKV(items)
    # use regex expression to obtain model words from title
    pattern = re.compile("[a-zA-Z0-9]*(([0-9]+[^0-9^,^]+)|([^0-9^,^]+[0-9]+))[a-zA-Z0-9]*")
    for product, title in dictionary.items():
        # gives set of all model words
        for match in pattern.finditer(title):
            word = match.group()
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            # word = word.replace(' ', '')
            if word not in modelWords_title:
                modelWords_title.append(word)

    pattern = re.compile("(\d+\.?\d*?[a-zA-Z]+$)")
    pattern2 = re.compile("(\d+\.?\d*)")
    for value in process.values():
        # gives set of all model words
        for match in pattern.finditer(value):
            word = match.group()
            if word.startswith(' '):
                word = word[1:]
            if word.endswith('kg'):
                word = word[:-2]
                value = str(float(word) * 2.2) + str('lbs')
            if word.endswith('lbsss'):
                word = word[:-2]
            if word.endswith('lbss'):
                word = word[:-1]
            if word.endswith('lb'):
                word = word.replace(word, 'lbs')

            if word not in modelWords_value:
                modelWords_value.append(word)

        for match in pattern2.finditer(value):
            word = match.group()
            if word.startswith(' '):
                word = word[1:]

            if word.endswith('lbss'):
                word = word[:-1]
            if word.endswith('lbss'):
                word = word[:-1]

            if word.endswith('kg'):
                word = word[:-2]
                value = str(float(word) * 2.2) + str('lbs')
                word = word.replace(word, value)
            if word.endswith('lb'):
                word = word.replace(word, 'lbs')

            if word not in modelWords_value:
                modelWords_value.append(word)

    modelWords = modelWords_title + modelWords_value
    binaryMatrix = np.full((len(modelWords), len(dictionary)), 0)

    values = []
    for value in process.values():
        values.append(value)

    for product, title in dictionary.items():
        # gives a binary vector for each product
        count = 0
        for word in modelWords:
            if word in title or word in values:
                binaryMatrix[count][product] = 1
            else:
                binaryMatrix[count][product] = 0
            count += 1

    return binaryMatrix





# Returns the signaturematrix

def minhash(binaryVector,prime):
    matrix = binaryVector
    numberRows, numberProducts = binaryVector.shape
    numHash = int(numberRows / 2)
    signature_matrix = np.full((numHash, numberProducts), np.inf)
    hash_functions = []
    # Set seed
    # generate the signature matrix using minhashing
    for row in tqdm(range(numberRows)):
        hash_row = []
        for i in range(numHash):
            int1 = random.randint(0, 2 ** 32 - 1)
            int2 = random.randint(0, 2 ** 32 - 1)
            hash_value = (int1 + int2 * (row + 1)) % prime
            hash_row.append(hash_value)
        hash_functions.append(hash_row)
        for column in range(numberProducts):
            if matrix[row][column] == 0:
                continue
            for i in range(numHash):
                value = hash_functions[row][i]
                if signature_matrix[i][column] > value:
                    signature_matrix[i][column] = value
    return signature_matrix




def jaccard_similarity(arr1, arr2):
    intersection = np.sum(np.logical_and(arr1, arr2))
    union = np.sum(np.logical_or(arr1, arr2))
    return intersection / union

# Locality Sensitive Hashing
def localitySensitiveHashing(signatureMatrix,binaryVector, b):
    n, d = signatureMatrix.shape
    r = int(n / b)
    threshold = math.pow((1 / b), (1 / r))
    print('The threshold is ' + str(threshold))
    hash_buckets = collections.defaultdict(set)
    similar = set()
    matrix = binaryVector
    bands = np.array_split(signatureMatrix, b, axis=0)
    for i, band in enumerate(bands):
        for j in range(d):
            band_id = tuple(list(band[:, j]) + [str(i)])
            hash_buckets[band_id].add(j)
    candidate_pairs = set()
    for bucket in hash_buckets.values():
        if len(bucket) > 1:
            for pair in combinations(bucket, 2):
                candidate_pairs.add(pair)
    for candidate in candidate_pairs:
        value_1 = int(candidate[0])
        value_2 = int(candidate[1])
        sig1 = (matrix[value_1])
        sig2 = (matrix[value_2])
        jaccardSim = jaccard_similarity(sig1,sig2)
        if jaccardSim >= threshold:
            similar.add(tuple(sorted((value_1, value_2))))
    return similar, candidate_pairs


# Returns the modelIDs of products
def getID(productDescription):
    modelID = []
    for product in productDescription:
        value = product['modelID']
        modelID.append(value)

    return modelID




# Returns a list of the true duplicates
def getDuplicate(productDescription):
    modelID = getID(productDescription)
    dictionaryID = getDictionary(modelID)
    duplicates = []
    for producti, idi in dictionaryID.items():
        for productj, idj in dictionaryID.items():
            if producti == productj:
                continue
            if idi == idj and [productj, producti] not in duplicates:
                duplicates.append([producti, productj])
    return duplicates


# The duplicates detected bij LSH
def getPredict(candidates, item_list):
    counter = 0
    true_duplicate = getDuplicate(item_list)
    for pair in candidates:
        for duplicate in true_duplicate:
            if pair[0] == duplicate[0] and pair[1] == duplicate[1]:
                counter += 1
    return counter

# Calculates pair completeness
def pairCompleteness(candidates, item_list):
    # precision = 1, so pair completeness = recall
    # = duplicates found / total number of duplicate
    duplicates = getDuplicate(item_list)
    totalDup = len(duplicates)
    foundDup = getPredict(candidates, item_list)
    pairCom = foundDup / totalDup

    return pairCom





# Calculates pair quality
def pairQuality(candidates, item_list):
    # the average number of duplicates found per comparison.
    # values on interval [0,2]
    numComparisons = len(candidates)  # candidate pairs proposed by LSH
    foundDup = getPredict(candidates, item_list)  # candidate pairs that are true duplicates
    pairQ = foundDup / numComparisons
    return pairQ


# Calculates the fraction of comparisons
def fracComparisons(candidates, item_list):
    numCom = len(candidates)
    # total number of possible comparisons
    # where total number of possible comparisons is combinations of 2 from the number of products
    n = len(item_list)
    totalCom = math.pow(n, 2)
    fraction = numCom / totalCom
    return fraction


# Calculates F1_star
def starmeasure(pairQuality, pairCompleteness):
    # harmonic mean between pair quality and pair completeness
    # value between one and zero
    PQ = pairQuality
    PC = pairCompleteness
    if PQ == 0 and PC == 0:
        F1_star = 0
    else:
        F1_star = (2 * PQ * PC) / (PQ + PC)
    return F1_star




# Returns the shop per product
def getShop(productDescription):
    shops = []
    n = len(productDescription)
    for product in productDescription:
        value = product['shop']
        shops.append(value)
    keys = [i for i in range(0, n)]
    shopDictionary = {keys[i]: shops[i] for i in range(len(keys))}
    return shopDictionary


# Process brand names
def preProcessBrandlist(brandlist):
    for i, brand in enumerate(brandlist):
        if type(brand) == str:
            brandlist[i] = brand.lower()
        elif brand == 'dynex\\x99':
            brandlist[i] = 'dynex'
    return brandlist


# Returns the brand per product
def getBrand(productDescription):
    path = 'C:/Users/chaya/OneDrive/Documenten/Master BAQM/Computer Science for Business Analytics/Paper/TVbrands.csv'
    df = pd.read_csv(path)
    brands = df['brands'].tolist()
    brands = preProcessBrandlist(brands)
    n = len(productDescription)
    brand = {}
    titles = getTitle(productDescription)
    titles = preProcessBrandlist(titles)

    for i, title in enumerate(titles):
        for element in brands:
            if element in title:
                brand.setdefault(i, []).append(element)
                break
        if title == 'newegg.com - 32" led 60hz 720p refurb':
            brand.setdefault(i, []).append('jvc')
        elif title == 'newegg.com - 50" eled 1080p 120hz':
            brand.setdefault(i, []).append('jvc')
        elif title == 'newegg.com - 47" led tv 1080p':
            brand.setdefault(i, []).append('lg')
        elif title == 'newegg.com - 60" plasma 3,000,000:1 1080p':
            brand.setdefault(i, []).append('lg')
        elif title == 'newegg.com - 55" eled 1080p 120hz':
            brand.setdefault(i, []).append('jvc')
        elif title == 'newegg.com - refurbished: 42" class 1080p pro:centric lcd hdtv with applications platform -42lt670h':
            brand.setdefault(i, []).append('lg')
        elif title == 'refurbished 47" class 47" diag. lcd tv 1080p hdtv 1080p e471vle - best buy':
            brand.setdefault(i, []).append('vizio')
        elif title == 'newegg.com - 42" led tv':
            brand.setdefault(i, []).append('lg')

    return brand


# key-value pairs met q similarity

# Q-gram functions
def qgram(string, q):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    Qgrams = zip(*[string[i:] for i in range(q)])
    list = [''.join(Qgram) for Qgram in Qgrams]

    return len(list)


def qgramDistance(s, r):
    if s == r:
        return 0
    elif len(s) == 1 or len(r) == 1:
        qgram = QGram(1)
        distance = qgram.distance(s, r)
    elif len(s) == 2 or len(r) == 2:
        qgram = QGram(2)
        distance = qgram.distance(s, r)
    else:
        qgram = QGram(3)
        distance = qgram.distance(s, r)
    return distance


def qgramSim(s, r):
    if len(s) == 1 or len(r) == 1:
        n1 = qgram(s, 1)
        n2 = qgram(r, 1)
        distance = qgramDistance(s, r)
        qgramSim = (n1 + n2 - distance) / (n1 + n2)
        return qgramSim
    elif len(s) == 2 or len(r) == 2:
        n1 = qgram(s, 2)
        n2 = qgram(r, 2)
        distance = qgramDistance(s, r)
        qgramSim = (n1 + n2 - distance) / (n1 + n2)
        return qgramSim
    else:
        n1 = qgram(s, 3)
        n2 = qgram(r, 3)
        distance = qgramDistance(s, r)
        qgramSim = (n1 + n2 - distance) / (n1 + n2)
    return qgramSim


# Another way to preprocess a string
def preprocess2(string):
    if type(string) == int:
        return string
    # remove round brackets, backslash, and -
    # change uppercase letters to lowercase
    for word in string:
        if type(word) == str:
            string = string.replace(word, word.lower())
    # normalize inch
    typeInch = [' inch', 'inches', '-inch', '"', 'inch']
    for word in typeInch:
        if word in string:
            string = string.replace(word, "inch")
    # normalize to hz
    typeHz = [' hz', 'hertz', '-hz']
    for word in typeHz:
        if word in string:
            string = string.replace(word, "hz")
    typeWeight = [' lbs', 'pounds', ' pounds', 'lb', ' lb', 'lbs.', ' lbs.']
    for word in typeWeight:
        if word in string:
            string = string.replace(word, "lbs")
    if string.endswith('lbss'):
        string = string[-2]
    if string.endswith('.'):
        string = string[-1]
    words = ['/', "(", ")", " -", " "]
    for word in words:
        if word in string:
            string = string.replace(word, "")

    return string


# exMW(p) all model words from the values of the attributes from product p;

def exMW(p, product,items):
    modelwords = []
    keyvalue = getKeyValue(items)
    pattern = re.compile(r"[a-zA-Z0-9]*(([0-9]+[^0-9^,]+)|([^0-9^,]+[0-9]+))[a-zA-Z0-9]*")
    for key in p:
        dictionary = keyvalue[product]
        find = dictionary[key]
        find = preprocess(find)
        for match in pattern.finditer(find):
            mw = match.group()
            modelwords.append(mw)
    return modelwords


# mw(C, D) percentage of matching model words from two sets of model words;
def mw(C, D):
    count = 0
    for mw_c in C:
        for mw_d in D:
            if mw_c == mw_d:
                count += 1
    total = len(C) + len(D)
    percentage = count / total

    return percentage


# TMWMSim(pi, pj , α, β) the TMWM similarity between the products i and j using the parameters α and β;
def TMWMSim(pi, pj, alpha, beta,processedTitle):
    # the cosine similarity of two product names is calculated.
    title_1 = processedTitle[pi]
    title_2 = processedTitle[pj]
    vectorizer = CountVectorizer().fit_transform([title_1, title_2])
    sim = cosine_similarity(vectorizer)

    cos_sim = sim[1][0]
    # A threshold alpha is used to determine if the considered products are duplicates.
    if cos_sim > alpha:
        similarity = cos_sim
        return similarity
    else:
        modelwords_1 = []
        modelwords_2 = []
        pattern = re.compile(r"[a-zA-Z0-9]*(([0-9]+[^0-9^,]+)|([^0-9^,]+[0-9]+))[a-zA-Z0-9]*")
        for match in pattern.finditer(title_1):
            word = match.group()
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word not in modelwords_1:
                modelwords_1.append(word)
        for match in pattern.finditer(title_2):
            word = match.group()
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word.startswith(' '):
                word = word[1:]
            if word not in modelwords_2:
                modelwords_2.append(word)

            dis = distance(modelwords_1, modelwords_2)
            # calculate the length of the longer string
            longer_length = max(len(modelwords_1), len(modelwords_2))
            # calculate the similarity
            sim = 1 - (dis / longer_length)
            if sim > 0.5:
                similarity = sim
                return similarity

        return -1


# minFeatures(pi, pj) the minimum of the number of product features that product i and j contain
def minFeatures(pi, pj,items):
    keyvalue = getKeyValue(items)
    features_1 = len(keyvalue[pi].keys())
    features_2 = len(keyvalue[pj].keys())
    minimum = min(features_1, features_2)
    return minimum

# Returns the dissimilarity matrix
def dissimilarityMatrix(items, lsh,processedTitle):
    n = len(items)
    # empty dissimilaritymatrix
    disMatrix = np.zeros((n, n))
    shops = getShop(items)
    brands = getBrand(items)
    keyvalue = getKeyValue(items)
    gamma = 0.756
    alpha = 0.602
    beta = 0.000
    mu = 0.650

    '''
    words = ['Screen Size Class', 'Screen Size', 'Width', 'Product Width', 'Screen Size (Measured Diagonally)',
         'TV Type',
         'Vertical Resolution', 'Recommended Resolution', 'Weight', 'Product Height (without stand)',
         'Maximum Resolution',
         'Product Weight', 'Manufacturer:', 'Diagonal Image Size:', 'Height:', 'Product Dimensions:',
         'Display Technology',
         'Display Size', 'Maximum Resolution', 'Shipping Weight']
    '''

    for product_i, shop_i in shops.items():
        for product_j, shop_j in shops.items():
            if shop_i == shop_j and product_i != product_j:
                disMatrix[product_i][product_j] = 100
            elif brands[product_i] != brands[product_j] and product_i != product_j:
                disMatrix[product_i][product_j] = 100
            elif (product_i, product_j) not in lsh and product_i != product_j:
                disMatrix[product_i][product_j] = 100
            elif product_i != product_j:
                sim = 0
                avgSim = 0
                # non-matching keys product i
                nmk_i = []
                nmk_j = []
                for keys in keyvalue[product_i].keys():
                    nmk_i.append(keys)
                # non-matching keys product j
                for keys in keyvalue[product_j].keys():
                    nmk_j.append(keys)
                # number of matches
                m = 0
                # weight of matches
                w = 0
                for keyi, valuei in keyvalue[product_i].items():
                    # if keyi in words:
                    for keyj, valuej in keyvalue[product_j].items():
                        # if keyj in words:
                        key_i = preprocess(keyi)
                        value_i = preprocess(valuei)
                        key_j = preprocess(keyj)
                        value_j = preprocess(valuej)
                        keySim = qgramSim(key_i, key_j)
                        if keySim > gamma:
                            valueSim = qgramSim(value_i, value_j)
                            weight = keySim
                            sim = sim + weight * valueSim
                            m = m + 1
                            w = w + weight
                            var_i = nmk_i.remove(keyi)
                            var_j = nmk_j.remove(keyj)
                if w > 0:
                    avgSim = sim / w
                mwPerc = mw(exMW(nmk_i, product_i,items), exMW(nmk_j, product_j,items))
                titleSim = TMWMSim(product_i, product_j, alpha, beta,processedTitle)
                if titleSim == -1:
                    theta_1 = m / minFeatures(product_i, product_j,items)
                    theta_2 = 1 - theta_1
                    hsim = theta_1 * avgSim + theta_2 * mwPerc
                else:
                    theta_1 = (1 - mu) * (m / minFeatures(product_i, product_j,items))
                    theta_2 = 1 - mu - theta_1
                    hsim = theta_1 * avgSim + theta_2 * mwPerc + mu * titleSim
                dissimilarity = 1 - hsim
                disMatrix[product_i][product_j] = dissimilarity

    return disMatrix




# Clustering
def clustering_single(disMatrix,epsilon):
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='single',
                                         distance_threshold=epsilon, compute_full_tree=True).fit(disMatrix)
    # Fit the model to the data
    labels = clustering.fit_predict(disMatrix)

    return labels



# Returns F1 score
def F1measure(item_list, cluster_labels):
    # harmonic mean between precision and recall
    duplicates = getDuplicate(item_list)
    count = 0
    dup = []
    # Print the cluster labels for each product
    for i, label in enumerate(cluster_labels):
        # print("Product ", i, ": Cluster ", label)
        for j, label2 in enumerate(cluster_labels):
            if i != j and label == label2 and label!=-1 and [j, i] not in dup:
                dup.append([i, j])
    for duplicate in dup:
        if duplicate in duplicates:
            count+= 1
    if count == 0:
        F1 = 0
    else:
        true_pos = count
        false_pos = len(dup) - true_pos
        precision = true_pos / (true_pos + false_pos)
        false_neg = len(duplicates) - true_pos
        recall = true_pos / (true_pos + false_neg)
        F1 = (2 * precision * recall) / (precision + recall)
    return F1

def clustering_complete(disMatrix,epsilon):
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='complete',
                                         distance_threshold=epsilon, compute_full_tree=True).fit(disMatrix)
    # Fit the model to the data
    labels = clustering.fit_predict(disMatrix)

    return labels

def clustering_DBSCAN(disMatrix,epsilon):
    clustering = DBSCAN(eps=epsilon,min_samples=2, metric='precomputed').fit(disMatrix)
    # Fit the model to the data
    labels = clustering.fit_predict(disMatrix)

    return labels



'''
# for testing
titles = getTitle(test_items)
processedTitle = preProcess(titles)
dictionary = getDictionary(processedTitle)
binaryProduct = getBinaryVector(dictionary,test_items)
signatureMatrix = minhash(binaryProduct,prime = 857)
lsh, candidates = localitySensitiveHashing(signatureMatrix,binaryProduct,b=80)
#modelIDs = getID(test_items)
PC = pairCompleteness(candidates, test_items)
print(PC)
PQ = pairQuality(candidates, test_items)
print(PQ)
print(fracComparisons(candidates, test_items))
print(starmeasure(PQ, PC))
disMatrix = dissimilarityMatrix(test_items, lsh,processedTitle)
cluster_labels = clustering(disMatrix,epsilon=0.5)
F1 = F1measure(test_items,cluster_labels)
print("F1 is" +str(F1))

# for training
titles = getTitle(train_items)
processedTitle = preProcess(titles)
dictionary = getDictionary(processedTitle)
binaryProduct = getBinaryVector(dictionary,train_items)
signatureMatrix = minhash(binaryProduct,prime = 1327)
lsh, candidates = localitySensitiveHashing(signatureMatrix,binaryProduct,b=80)
#modelIDs = getID(test_items)
PC = pairCompleteness(candidates, train_items)
print(PC)
PQ = pairQuality(candidates, train_items)
print(PQ)
print(fracComparisons(candidates, train_items))
print(starmeasure(PQ, PC))
disMatrix = dissimilarityMatrix(train_items, lsh,processedTitle)
cluster_labels = clustering(disMatrix,epsilon=0.5)
F1 = F1measure(train_items,cluster_labels)
print("F1 is" +str(F1))
'''