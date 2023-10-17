import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
path="D:/OneDrive - Northeastern University/NEU/NLP/HW1/output.pkl"

pkl_dir = open(path, 'rb') #opening the pkl file
data = pickle.load(pkl_dir) #loading the pkl file to read the model parameters from the trained data
# close the file
pkl_dir.close() 

logprior = data[0]
loglikelihood = data[1]

def remove_stopwords(text):
    temp=[]
    for word in text.split():
        if word in stopwords.words('english'):
            temp.append('')
        else:
            temp.append(word)
    x=temp[:]
    temp.clear()
    return " ".join(x)

def clean_review(review):
    review_cleaned=review.lower() #converting the reviews to lowercase
    review_cleaned=re.sub(r'<.*?>','',review_cleaned)  #the html tags are replaced with the empty strings
    review_cleaned=re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', review_cleaned) #removing URL's form the text
    #removing punctuations form the text
    exclude=string.punctuation
    review_cleaned=review_cleaned.translate(str.maketrans('','',exclude))
    review_cleaned=remove_stopwords(review_cleaned) #calling an external function that removes the stopwords from each review.
    ps=PorterStemmer() #initializing the porter stemmer object
    review_cleaned=" ".join([ps.stem(word) for word in review_cleaned.split()])
    return review_cleaned


def naive_bayes_predict(review, logprior, loglikelihood):
    '''
    Params:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Return:
        total_prob: the sum of all the loglikelihoods of each word in the review (if found in the dictionary) + logprior (a number)

    '''  
      # process the review to get a list of words
    word_l = clean_review(review)
    word_l=word_l.split()
    # initialize probability to zero
    total_prob = 0
    # add the logprior
    total_prob += logprior
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            total_prob = total_prob + loglikelihood[word]
    
    if total_prob<0:
        total_prob=1
    else:
        total_prob=0
        
    return total_prob

if __name__=='__main__':
    a=True
    while a:
        user_review = input("enter the review here or X to stop the process : ")
        if user_review !='X':
            prediction = naive_bayes_predict(user_review, logprior, loglikelihood)
            if(prediction == 1):
                print("review is negative")
            else:
                print("review is positive")
        else:
            a=False

