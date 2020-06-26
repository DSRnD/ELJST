"""
(c) Ayan Sengupta - 2020
License: MIT License

Implementation of JST (Joint Sentiment-Topic model)

Reference
    [1] https://www.researchgate.net/publication/47454505_oroopenacuk_A_Comparative_Study_of_Bayesian_Models_for_Unsupervised_Sentiment_Detection

"""

# Author: Ayan Sengupta

#import warnings
#from collections import defaultdict
#import inspect
from __future__ import absolute_import
from sklearn.utils.validation import check_is_fitted, check_non_negative, check_random_state, check_array
import numpy as np
import scipy
from scipy.special import gammaln, psi
from .base import BaseEstimator
from .utils import sampleFromDirichlet, sampleFromCategorical, log_multi_beta, word_indices
from .utils import coherence_score_uci, coherence_score_umass, symmetric_kl_score, Hscore

class JST(BaseEstimator):
    """JST model
    
    Parameters
    ----------
    n_topic_components : int, optional (default=10)
        Number of topics.
    n_sentiment_components : int, optional (default=5)
        Number of sentiments.
    doc_sentiment_prior : float, optional (default=None)
        Prior of document sentiment distribution `theta`. If the value is None,
        defaults to `1 / n_sentiment_components`.
    doc_sentiment_topic_prior : float, optional (default=None)
        Prior of document topic-sentiment distribution `pi`. If the value is None,
        defaults to `1 / n_topic_components`.
    topic_sentiment_word_prior : float, optional (default=None)
        Prior of topic-sentiment word distribution `beta`. If the value is None, defaults
        to `1 / (n_topic_components * n_sentiment_components)`.
    max_iter : integer, optional (default=10)
        The maximum number of iterations for Gibbs sampling.
    prior_update_step: integer, optional (default=5)
        How often to update priors using Minka's fixed point iteration
    evaluate_every : int, optional (default=0)
        How often to evaluate perplexity. Only used in `fit` method.
        set it to 0 or negative number to not evaluate perplexity in
        training at all. Evaluating perplexity can help you check convergence
        in training process, but it will also increase total training time.
        Evaluating perplexity in every iteration might increase training time
        up to two-fold.
    verbose : int, optional (default=0)
        Verbosity level.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    components_ : array, [vocabSize, n_topic_components, n_sentiment_components]
        topic-sentiment word distribution. Since the complete
        conditional for topic word distribution is a Dirichlet,
        ``components_[i, j, k]`` can be viewed as pseudocount that represents the
        number of times word `i` was assigned to topic `j` and sentiment `k`.
        It can also be viewed as distribution over the words for each topic-sentiment pair
        after normalization:
        ``model.components_ / model.components_.sum(axis=0)[np.newaxis,:,:]``.
    doc_sentiment_prior_ : float
        Prior of document sentiment distribution `theta`. If the value is None,
        it is `1 / n_sentiment_components`.
    doc_sentiment_topic_prior_ : float
        Prior of document-sentiment-topics distribution `pi`. If the value is None,
        it is `1 / n_topic_components`.
    topic_sentiment_word_prior_ : float
        Prior of topic-sentiment-word distribution `beta`. If the value is None, it is
        `1 / (n_topic_components * n_sentiment_components)`.
    Examples
    --------
    >>> from jointtsmodels.JST import JST
    >>> from jointtsmodels.utils import coherence_score_uci
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> from sklearn.datasets import fetch_20newsgroups
    >>> # This produces a feature matrix of token counts, similar to what
    >>> # CountVectorizer would produce on text.
    >>> data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
    >>> data = data[:1000]
    >>> vectorizer = CountVectorizer(max_df=0.7, min_df=10,
                                max_features=5000,
                                stop_words='english')
    >>> X = vectorizer.fit_transform(data)
    >>> vocabulary = vectorizer.get_feature_names()
    >>> inv_vocabulary = dict(zip(vocabulary,np.arange(len(vocabulary))))
    >>> lexicon_data = pd.read_excel('../lexicon/prior_sentiment.xlsx')
    >>> lexicon_data = lexicon_data.dropna()
    >>> lexicon_dict = dict(zip(lexicon_data['Word'],lexicon_data['Sentiment']))
    >>> model = JST(n_topic_components=5,n_sentiment_components=5,
    ...     random_state=0)
    >>> model.fit(X.toarray(), lexicon_dict)
    JST(...)
    >>> # get topics for some given samples:
    >>> model.transform()[:2]
    array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
           [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])
    >>> top_words = list(model.getTopKWords(vocabulary).values())
    >>> coherence_score_uci(X.toarray(),inv_vocabulary,top_words)
    1.107204574754555
           
    Reference
    ---------
        [1] https://www.researchgate.net/publication/47454505_oroopenacuk_A_Comparative_Study_of_Bayesian_Models_for_Unsupervised_Sentiment_Detection
    
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    def __init__(self, n_topic_components=10, n_sentiment_components=5, doc_topic_prior=None, doc_sentiment_prior=None,
                 doc_topic_sentiment_prior=None, doc_sentiment_topic_prior=None,
                 topic_sentiment_word_prior=None, max_iter=10,
                 prior_update_step=5, evaluate_every=1, verbose=1, random_state=None):
        
        super().__init__(n_topic_components=n_topic_components, n_sentiment_components=n_sentiment_components, doc_topic_prior=doc_topic_prior, doc_sentiment_prior=doc_sentiment_prior,
                 doc_topic_sentiment_prior=doc_topic_sentiment_prior, doc_sentiment_topic_prior=doc_sentiment_topic_prior,
                 topic_sentiment_word_prior=topic_sentiment_word_prior, max_iter=max_iter,
                 prior_update_step=prior_update_step, evaluate_every=evaluate_every, verbose=verbose, random_state=random_state)
        
    def _initialize_(self, X, lexicon_dict):
        """Initialize fit variables
        Parameters
        ----------
        X : array-like, shape=(n_docs, vocabSize)
            Document word matrix.
        lexicon_dict : dict
            Dictionary of word lexicons with sentiment score
        Returns
        -------
        self
        """
        
        self.wordOccurenceMatrix = X
        self._check_params()
        self._init_latent_vars()
        
        n_docs, vocabSize = self.wordOccurenceMatrix.shape

        # Pseudocounts
        self.n_ds = np.zeros((n_docs, self.n_sentiment_components))
        self.n_dst = np.zeros((n_docs, self.n_sentiment_components,self.n_topic_components))
        self.n_d = np.zeros((n_docs))
        self.n_vts = np.zeros((vocabSize, self.n_topic_components, self.n_sentiment_components))
        self.n_ts = np.zeros((self.n_topic_components, self.n_sentiment_components))

        self.topics = {}
        self.sentiments = {}

        self.alphaVec = self.doc_sentiment_prior_.copy()
        self.gammaVec = self.doc_sentiment_topic_prior_
        self.beta = self.topic_sentiment_word_prior_

        for d in range(n_docs):
            
            sentimentDistribution = sampleFromDirichlet(self.alphaVec)
            topicDistribution = np.zeros(( self.n_sentiment_components,self.n_topic_components))
            for s in range(self.n_sentiment_components):
                topicDistribution[s, :] = sampleFromDirichlet(self.gammaVec)
            for i, w in enumerate(word_indices(self.wordOccurenceMatrix[d, :])):
               
                   s = sampleFromCategorical(sentimentDistribution)
                   t = sampleFromCategorical(topicDistribution[s, :])
                  
                   prior_sentiment = lexicon_dict.get(w,1)
                   
                   self.topics[(d, i)] = t
                   self.sentiments[(d, i)] = s
                   self.n_ds[d,s]+=1
                   self.n_dst[d,s,t] += 1
                   self.n_d[d] += 1
                   self.n_vts[w, t, s*prior_sentiment] += 1
                   self.n_ts[t, s] += 1
    
    def conditionalDistribution(self, d, v):
        """
        Calculates the joint topic-sentiment probability for word v in document d
        Parameters
        -----------
        d: index
            Document index
        v: index
            Word index
        Returns
        ------------
        x: matrix
            Matrix (n_topic_components x n_sentiment_components) of joint probabilities
        """
        probabilities_ts = np.ones((self.n_topic_components, self.n_sentiment_components))
        firstFactor = (self.n_ds[d] + self.alphaVec) / \
            (self.n_d[d] +  np.sum(self.alphaVec))
        secondFactor = np.zeros((self.n_topic_components,self.n_sentiment_components))
        for s in range(self.n_sentiment_components):
        
             secondFactor[:,s] = ((self.n_dst[d, s, :] + self.gammaVec) / \
            (self.n_ds[d, s] + np.sum(self.gammaVec)))

        thirdFactor = (self.n_vts[v,:, :] + self.beta) / \
            (self.n_ts + self.n_vts.shape[0] * self.beta)

        probabilities_ts *= firstFactor[np.newaxis,:]
        probabilities_ts *= secondFactor * thirdFactor
        probabilities_ts /= np.sum(probabilities_ts)
        
        return probabilities_ts
        
    def fit(self, X, lexicon_dict, rerun=False, max_iter=None):
        """Learn model for the data X with Gibbs sampling.
        Parameters
        ----------
        X : array-like, shape=(n_docs, vocabSize)
            Document word matrix.
        lexicon_dict : dict
            Dictionary of word lexicons with sentiment score
        rerun: bool (default=False)
            If True then we do not re initialize the model
        max_iter : int, optional (default=None)
        Returns
        -------
        self
        """
        if rerun == False:
            self._initialize_(X, lexicon_dict)
            
        self.wordOccurenceMatrix = self._check_non_neg_array(self.wordOccurenceMatrix, "JST.fit")
        if max_iter is None:
            max_iter = self.max_iter
        
        self.all_loglikelihood = []
        self.all_perplexity = []
        n_docs, vocabSize = self.wordOccurenceMatrix.shape
        for iteration in range(max_iter):
            for d in range(n_docs):
                for i, v in enumerate(word_indices(self.wordOccurenceMatrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    prior_sentiment = lexicon_dict.get(v,1)
                    self.n_ds[d,s]-=1
                    self.n_d[d] -= 1
                    self.n_dst[d,s,t] -= 1
                    self.n_vts[v, t, s*prior_sentiment] -= 1
                    self.n_ts[t, s] -= 1

                    probabilities_ts = self.conditionalDistribution(d, v)
                    ind = sampleFromCategorical(probabilities_ts.flatten())
                    t, s = np.unravel_index(ind, probabilities_ts.shape)
                    
                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_d[d] += 1
                    self.n_dst[d,s,t] += 1
                    self.n_vts[v, t, s*prior_sentiment] += 1
                    self.n_ts[t, s] += 1
                    self.n_ds[d,s]+=1
            
            if self.prior_update_step > 0 and (iteration+1)%self.prior_update_step == 0:
                numerator = 0
                denominator = 0
                for d in range(n_docs):
                    numerator += psi(self.n_d[d] + self.alphaVec) - psi(self.alphaVec)
                    denominator += psi(np.sum(self.n_ds[d] + self.alphaVec)) - psi(np.sum(self.alphaVec))
                
                self.alphaVec *= numerator / denominator     

            loglikelihood_ = self.loglikelihood()
            perplexity_ = self.perplexity()
            
            self.all_loglikelihood.append(loglikelihood_)
            self.all_perplexity.append(perplexity_)
            
            if self.evaluate_every > 0 and (iteration+1)%self.evaluate_every == 0:
                if self.verbose > 0:
                    print ("Perplexity after iteration {} (out of {} iterations) is {:.2f}".format(iteration + 1, max_iter, perplexity_))
        
        self.doc_sentiment_prior_ = self.alphaVec
        normalized_n_vts = self.n_vts.copy() + self.beta
        normalized_n_vts /= normalized_n_vts.sum(0)[np.newaxis,:,:]
        self.components_ = normalized_n_vts
        
        return self
        
    def _unnormalized_transform(self):
        """Transform data according to fitted model.
        Returns
        -------
        doc_sentiment_distr : shape=(n_docs, n_sentiment_components)
            Document sentiment distribution for X.
        """
        return self.n_ds + self.doc_sentiment_prior_
        
    def transform(self):
        """Transform data according to fitted model.
        Returns
        -------
        doc_sentiment_distr : shape=(n_docs, n_sentiment_components)
            Document sentiment distribution for X.
        """
        normalize_n_ds = self._unnormalized_transform().copy()
        normalize_n_ds /= normalize_n_ds.sum(1)[:,np.newaxis]
        return normalize_n_ds

    def fit_transform(self, X, lexicon_dict, rerun=False, max_iter=None):
        """Fit and transform data according to fitted model.
        Parameters
        ----------
        X : array-like, shape=(n_docs, vocabSize)
            Document word matrix.
        lexicon_dict : dict
            Dictionary of word lexicons with sentiment score
        rerun: bool (default=False)
            If True then we do not re initialize the model
        max_iter : int, optional (default=None)
        Returns
        -------
        doc_sentiment_distr : shape=(n_samples, n_sentiment_components)
            Document sentiment distribution for X.
        """
        return self.fit(X, lexicon_dict, rerun=rerun, max_iter=max_iter).transform()
    
    def pi(self):
        """Document-sentiment-topic distribution according to fitted model.
        Returns
        -------
        doc_sentiment_topic_dstr : shape=(n_docs, n_sentiment_components, n_topic_components)
            Document-sentiment-topic distribution for X.
        """
        normalized_n_dst = self.n_dst.copy() + self.gammaVec
        normalized_n_dst /= normalized_n_dst.sum(2)[:,:,np.newaxis]
        return normalized_n_dst
        
    def loglikelihood(self):
        """Calculate log-likelihood of generating the whole corpus
        Returns
        -----------
        Log-likelihood score: float
        """
        n_docs, vocabSize = self.wordOccurenceMatrix.shape
        lik = 0

        for z in range(self.n_topic_components):
            for s in range(self.n_sentiment_components):
                lik += log_multi_beta(self.n_vts[:,z, s]+self.beta)
        
        lik -= self.n_topic_components * self.n_sentiment_components * log_multi_beta(self.beta, vocabSize)

        for m in range(n_docs):
            for z in range(self.n_sentiment_components):
                lik += log_multi_beta(self.n_dst[m, z, :]+self.gammaVec)
        
            lik += log_multi_beta(self.n_ds[m,:]+self.alphaVec)
        
        lik -= n_docs * self.n_sentiment_components * log_multi_beta(self.gammaVec)
        lik -= n_docs * log_multi_beta(self.alphaVec)
    
        return lik
        
    def perplexity(self):
        """Calculate approximate perplexity for the whole corpus.
        Perplexity is defined as exp(-1. * log-likelihood per word)
        
        Returns
        ------------
        score : float
        """
        score = np.exp(-self.loglikelihood()/self.wordOccurenceMatrix.sum())
        return score