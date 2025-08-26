import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
import random
import re
from scipy.spatial.distance import cosine
from scipy.stats import poisson, beta
import time
import requests
from bs4 import BeautifulSoup
import urllib.parse
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
import json
import pickle
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class IndianBuffetProcess:
    def __init__(self, alpha=2.0, sigma_x=1.0, sigma_a=1.0):
        """
        Indian Buffet Process for multi-feature document modeling
        
        Args:
            alpha: Concentration parameter (higher = more features)
            sigma_x: Noise variance for observations
            sigma_a: Prior variance for feature weights
        """
        self.alpha = alpha
        self.sigma_x = sigma_x
        self.sigma_a = sigma_a
        
        # IBP state variables
        self.Z = None  # Binary feature assignment matrix (N x K)
        self.A = None  # Feature weight matrix (K x D)
        self.n_features = 0
        self.feature_counts = None
        self.feature_names = {}
        
    def fit(self, X, n_iterations=200, verbose=True):
        """
        Fit IBP model using Gibbs sampling
        
        Args:
            X: Document-term matrix (N x D)
            n_iterations: Number of MCMC iterations
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        N, D = X.shape
        
        # Initialize IBP assignments
        self._initialize_ibp(N, D)
        
        # Gibbs sampling
        for iteration in range(n_iterations):
            # Sample existing feature assignments
            for n in range(N):
                for k in range(self.n_features):
                    self._sample_existing_feature(n, k, X)
            
            # Sample new features for each document
            for n in range(N):
                self._sample_new_features(n, X)
            
            # Sample feature weights
            self._sample_feature_weights(X)
            
            # Clean up unused features
            if iteration % 20 == 0:
                self._cleanup_features()
                if verbose and iteration % 40 == 0:
                    print(f"Iteration {iteration}: {self.n_features} features, "
                          f"avg features per doc: {np.mean(np.sum(self.Z, axis=1)):.2f}")
        
        if verbose:
            print(f"Final model: {self.n_features} latent features")
        
    def _initialize_ibp(self, N, D):
        """Initialize IBP with stick-breaking construction"""
        # First customer gets Poisson(alpha) dishes
        initial_features = poisson.rvs(self.alpha)
        initial_features = max(initial_features, 1)  # At least 1 feature
        
        self.n_features = initial_features
        self.Z = np.zeros((N, initial_features), dtype=int)
        self.feature_counts = np.zeros(initial_features)
        
        # First customer tries all initial features
        self.Z[0, :initial_features] = 1
        self.feature_counts[:initial_features] = 1
        
        # Subsequent customers
        for n in range(1, N):
            # Try existing features with probability proportional to popularity
            for k in range(self.n_features):
                prob = self.feature_counts[k] / (n + 1)
                if np.random.random() < prob:
                    self.Z[n, k] = 1
                    self.feature_counts[k] += 1
            
            # Try new features
            new_features = poisson.rvs(self.alpha / (n + 1))
            if new_features > 0:
                self._add_new_features(new_features, n)
        
        # Initialize feature weights
        self.A = np.random.normal(0, self.sigma_a, (self.n_features, D))
    
    def _add_new_features(self, n_new, customer_idx):
        """Add new features to the model"""
        if n_new <= 0:
            return
        
        old_features = self.n_features
        self.n_features += n_new
        
        # Expand Z matrix
        N = self.Z.shape[0]
        new_Z = np.zeros((N, self.n_features), dtype=int)
        new_Z[:, :old_features] = self.Z
        new_Z[customer_idx, old_features:] = 1
        self.Z = new_Z
        
        # Expand feature counts
        new_counts = np.zeros(self.n_features)
        new_counts[:old_features] = self.feature_counts
        new_counts[old_features:] = 1
        self.feature_counts = new_counts
        
        # Expand A matrix
        new_A = np.random.normal(0, self.sigma_a, (n_new, self.A.shape[1]))
        self.A = np.vstack([self.A, new_A])
    
    def _sample_existing_feature(self, n, k, X):
        """Sample assignment of document n to feature k"""
        # Remove current assignment
        old_assignment = self.Z[n, k]
        self.Z[n, k] = 0
        self.feature_counts[k] -= old_assignment
        
        # Calculate probabilities for z_nk = 0 and z_nk = 1
        log_prob_0 = self._log_likelihood_for_assignment(n, k, 0, X)
        log_prob_1 = self._log_likelihood_for_assignment(n, k, 1, X)
        
        # Prior probabilities
        N = self.Z.shape[0]
        prob_1_prior = self.feature_counts[k] / N
        prob_0_prior = 1 - prob_1_prior
        
        # Combine with likelihood
        log_prob_1 += np.log(prob_1_prior + 1e-10)
        log_prob_0 += np.log(prob_0_prior + 1e-10)
        
        # Normalize and sample
        max_log_prob = max(log_prob_0, log_prob_1)
        prob_1 = np.exp(log_prob_1 - max_log_prob)
        prob_0 = np.exp(log_prob_0 - max_log_prob)
        
        prob_1 = prob_1 / (prob_1 + prob_0)
        
        new_assignment = 1 if np.random.random() < prob_1 else 0
        self.Z[n, k] = new_assignment
        self.feature_counts[k] += new_assignment
    
    def _sample_new_features(self, n, X):
        """Sample new features for document n"""
        # Probability of new features follows Poisson(alpha/N)
        N = self.Z.shape[0]
        n_new = poisson.rvs(self.alpha / N)
        
        if n_new > 0:
            # For each potential new feature, calculate if it should be activated
            for _ in range(n_new):
                # Calculate likelihood improvement with new feature
                log_prob_with = self._log_likelihood_new_feature(n, X, active=True)
                log_prob_without = self._log_likelihood_new_feature(n, X, active=False)
                
                prob_active = 1 / (1 + np.exp(log_prob_without - log_prob_with))
                
                if np.random.random() < prob_active:
                    self._add_new_features(1, n)
    
    def _log_likelihood_for_assignment(self, n, k, assignment, X):
        """Calculate log-likelihood for a specific assignment"""
        # Predict document n with and without feature k
        prediction = np.dot(self.Z[n], self.A)
        
        # Modify prediction based on assignment
        if assignment == 1:
            if self.Z[n, k] == 0:  # Adding feature
                prediction += self.A[k]
        else:  # assignment == 0
            if self.Z[n, k] == 1:  # Removing feature
                prediction -= self.A[k]
        
        # Gaussian likelihood
        residual = X[n] - prediction
        log_likelihood = -0.5 * np.sum(residual**2) / (self.sigma_x**2)
        
        return log_likelihood
    
    def _log_likelihood_new_feature(self, n, X, active):
        """Calculate likelihood for adding a new feature"""
        if not active:
            return 0.0
        
        # Sample new feature weights
        D = X.shape[1]
        new_weights = np.random.normal(0, self.sigma_a, D)
        
        # Current prediction
        current_pred = np.dot(self.Z[n], self.A)
        
        # Prediction with new feature
        new_pred = current_pred + new_weights
        
        # Likelihood improvement
        current_residual = X[n] - current_pred
        new_residual = X[n] - new_pred
        
        log_likelihood_current = -0.5 * np.sum(current_residual**2) / (self.sigma_x**2)
        log_likelihood_new = -0.5 * np.sum(new_residual**2) / (self.sigma_x**2)
        
        return log_likelihood_new - log_likelihood_current
    
    def _sample_feature_weights(self, X):
        """Sample feature weights A given assignments Z"""
        N, D = X.shape
        
        for k in range(self.n_features):
            if self.feature_counts[k] == 0:
                continue
            
            # Documents that have this feature
            docs_with_feature = self.Z[:, k] == 1
            
            if not np.any(docs_with_feature):
                continue
            
            # Residual when removing this feature's contribution
            residual = X - np.dot(self.Z, self.A)
            residual += np.outer(self.Z[:, k], self.A[k])  # Add back this feature's contribution
            
            # Update weights for this feature
            # Posterior is Gaussian due to conjugacy
            precision = self.feature_counts[k] / (self.sigma_x**2) + 1 / (self.sigma_a**2)
            
            for d in range(D):
                mean_numerator = np.sum(self.Z[docs_with_feature, k] * residual[docs_with_feature, d]) / (self.sigma_x**2)
                mean = mean_numerator / precision
                variance = 1 / precision
                
                self.A[k, d] = np.random.normal(mean, np.sqrt(variance))
    
    def _cleanup_features(self):
        """Remove unused features"""
        active_features = self.feature_counts > 0
        
        if not np.all(active_features):
            self.Z = self.Z[:, active_features]
            self.A = self.A[active_features]
            self.feature_counts = self.feature_counts[active_features]
            self.n_features = np.sum(active_features)
    
    def get_feature_activations(self):
        """Get feature activation matrix"""
        return self.Z.copy()
    
    def get_feature_weights(self):
        """Get feature weight matrix"""
        return self.A.copy()
    
    def predict(self, doc_idx):
        """Predict document representation"""
        return np.dot(self.Z[doc_idx], self.A)

class AdvancedWikipediaScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'IBP-Research-Tool/1.0 (Educational Research Purpose)'
        })
        self.base_url = "https://en.wikipedia.org"
        
    def search_articles(self, query, limit=100):
        """Enhanced search with better filtering"""
        search_url = f"{self.base_url}/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'srprop': 'size|wordcount|timestamp'
        }
        
        try:
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            if 'query' in data and 'search' in data['query']:
                for result in data['query']['search']:
                    # Filter out disambiguation and list pages
                    title = result['title']
                    if not any(skip in title.lower() for skip in ['disambiguation', 'list of', 'category:']):
                        articles.append({
                            'title': title,
                            'pageid': result['pageid'],
                            'size': result.get('size', 0),
                            'wordcount': result.get('wordcount', 0)
                        })
            
            return articles
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    def get_article_content(self, title, max_chars=2500):
        """Get article content with better length control"""
        try:
            content_url = f"{self.base_url}/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'exintro': False,
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(content_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pages = data['query']['pages']
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    text = page_data['extract']
                    text = self.clean_text(text)
                    
                    # Intelligent truncation
                    if len(text) > max_chars:
                        # Split into sentences and take complete sentences
                        sentences = re.split(r'[.!?]+', text)
                        selected_text = ""
                        
                        for sentence in sentences:
                            if len(selected_text) + len(sentence) > max_chars:
                                break
                            selected_text += sentence + ". "
                        
                        return selected_text.strip()
                    
                    return text
            
            return None
        except Exception as e:
            print(f"Error getting content for '{title}': {e}")
            return None
    
    def clean_text(self, text):
        """Enhanced text cleaning"""
        if not text:
            return ""
        
        # Remove Wikipedia artifacts
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

def scrape_multi_topic_documents(target_count=3500):
    """Scrape Wikipedia documents with focus on multi-topic articles"""
    
    scraper = AdvancedWikipediaScraper()
    
    # Define single-topic searches (from original)
    single_topics = [
        "cricket sport",
        "basketball sport", 
        "machine learning artificial intelligence",
        "number theory mathematics",
        "climate change environmental",
        "genomics molecular biology",
        "Indian politics government",
        "Indian classical music",
        "opera classical music",
        "jazz bebop music",
        "botany plant biology",
        "mountains geography geology"
    ]
    
    # Define multi-topic searches for IBP
    multi_topics = [
        # Sports + Politics
        "athlete activism political",
        "sports politics government policy",
        "Olympic politics international relations",
        "FIFA corruption politics",
        "sports diplomacy international",
        
        # Climate + Politics  
        "climate change policy politics",
        "environmental politics green party",
        "carbon tax policy government",
        "climate activism political movement",
        "environmental regulation policy",
        
        # Meat + Environment
        "meat consumption environmental impact",
        "livestock agriculture climate change",
        "vegetarianism environmental ethics",
        "factory farming environmental",
        "sustainable agriculture meat production",
        
        # Sports Fitness + Performance
        "athlete performance sports science",
        "sports nutrition fitness training",
        "exercise physiology athletic performance",
        "sports medicine fitness health",
        "athletic training scientific methods"
    ]
    
    all_articles = []
    scraped_titles = set()
    
    print("Scraping Wikipedia articles for IBP analysis...")
    
    # Allocate articles across categories
    single_topic_target = target_count // 2  # 1750
    multi_topic_target = target_count - single_topic_target  # 1750
    
    articles_per_single_topic = single_topic_target // len(single_topics)
    articles_per_multi_topic = multi_topic_target // len(multi_topics)
    
    # Scrape single-topic articles
    print(f"Scraping single-topic articles ({articles_per_single_topic} per topic)...")
    for topic in single_topics:
        if len(all_articles) >= target_count:
            break
            
        print(f"  Scraping: {topic}")
        search_results = scraper.search_articles(topic, limit=articles_per_single_topic + 30)
        
        topic_count = 0
        for article in search_results:
            if len(all_articles) >= target_count or topic_count >= articles_per_single_topic:
                break
                
            title = article['title']
            if title in scraped_titles or article.get('wordcount', 0) < 100:
                continue
                
            content = scraper.get_article_content(title)
            if content and len(content) > 300:
                all_articles.append({
                    'title': title,
                    'content': content,
                    'topic_category': 'single',
                    'primary_topic': topic.split()[0],
                    'search_query': topic,
                    'source': 'wikipedia'
                })
                scraped_titles.add(title)
                topic_count += 1
            
            time.sleep(0.1)
        
        print(f"    Scraped {topic_count} articles")
    
    # Scrape multi-topic articles  
    print(f"Scraping multi-topic articles ({articles_per_multi_topic} per topic)...")
    for topic in multi_topics:
        if len(all_articles) >= target_count:
            break
            
        print(f"  Scraping: {topic}")
        search_results = scraper.search_articles(topic, limit=articles_per_multi_topic + 20)
        
        topic_count = 0
        for article in search_results:
            if len(all_articles) >= target_count or topic_count >= articles_per_multi_topic:
                break
                
            title = article['title']
            if title in scraped_titles or article.get('wordcount', 0) < 100:
                continue
                
            content = scraper.get_article_content(title)
            if content and len(content) > 300:
                all_articles.append({
                    'title': title,
                    'content': content,
                    'topic_category': 'multi',
                    'primary_topic': 'multi_topic',
                    'search_query': topic,
                    'source': 'wikipedia'
                })
                scraped_titles.add(title)
                topic_count += 1
            
            time.sleep(0.1)
        
        print(f"    Scraped {topic_count} articles")
    
    print(f"\nTotal articles scraped: {len(all_articles)}")
    
    # Extract content and metadata
    documents = [article['content'] for article in all_articles]
    metadata = [{
        'title': article['title'], 
        'topic_category': article['topic_category'],
        'primary_topic': article['primary_topic'],
        'search_query': article['search_query']
    } for article in all_articles]
    
    return documents, metadata

def extract_feature_interpretations(ibp_model, vectorizer, top_k=10):
    """Extract interpretable descriptions for IBP features"""
    
    feature_names = vectorizer.get_feature_names_out()
    feature_interpretations = {}
    
    for k in range(ibp_model.n_features):
        # Get feature weights
        weights = ibp_model.A[k]
        
        # Get top positive and negative weights
        top_pos_idx = np.argsort(weights)[-top_k:][::-1]
        top_neg_idx = np.argsort(weights)[:top_k]
        
        positive_terms = [(feature_names[i], weights[i]) for i in top_pos_idx if weights[i] > 0]
        negative_terms = [(feature_names[i], weights[i]) for i in top_neg_idx if weights[i] < 0]
        
        # Calculate feature activation stats
        feature_activation = ibp_model.Z[:, k]
        activation_rate = np.mean(feature_activation)
        n_active_docs = np.sum(feature_activation)
        
        feature_interpretations[k] = {
            'positive_terms': positive_terms,
            'negative_terms': negative_terms,
            'activation_rate': activation_rate,
            'n_active_docs': n_active_docs,
            'feature_strength': np.std(weights)
        }
    
    return feature_interpretations

def create_ibp_visualizations(ibp_model, metadata, feature_interpretations, documents):
    """Create comprehensive IBP visualizations"""
    
    # 1. Feature activation heatmap
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Heatmap of feature activations (sample)
    n_docs_display = min(100, ibp_model.Z.shape[0])
    sample_indices = np.random.choice(ibp_model.Z.shape[0], n_docs_display, replace=False)
    
    im1 = ax1.imshow(ibp_model.Z[sample_indices].T, cmap='Blues', aspect='auto')
    ax1.set_title(f'Feature Activation Matrix (Sample of {n_docs_display} docs)', fontsize=14)
    ax1.set_xlabel('Document Index')
    ax1.set_ylabel('Feature Index')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Feature activation rates
    activation_rates = [feature_interpretations[k]['activation_rate'] 
                       for k in range(ibp_model.n_features)]
    
    ax2.bar(range(len(activation_rates)), activation_rates)
    ax2.set_title('Feature Activation Rates', fontsize=14)
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('Activation Rate')
    ax2.set_ylim(0, 1)
    
    # 3. Documents per feature histogram
    docs_per_feature = [feature_interpretations[k]['n_active_docs'] 
                       for k in range(ibp_model.n_features)]
    
    ax3.hist(docs_per_feature, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_title('Distribution of Documents per Feature', fontsize=14)
    ax3.set_xlabel('Number of Active Documents')
    ax3.set_ylabel('Number of Features')
    
    # 4. Features per document histogram
    features_per_doc = np.sum(ibp_model.Z, axis=1)
    
    ax4.hist(features_per_doc, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.set_title('Distribution of Features per Document', fontsize=14)
    ax4.set_xlabel('Number of Active Features')
    ax4.set_ylabel('Number of Documents')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Feature co-occurrence network
    create_feature_network(ibp_model, feature_interpretations, top_features=15)

def create_feature_network(ibp_model, feature_interpretations, top_features=15):
    """Create network visualization of feature co-occurrences"""
    
    # Select most interesting features (by activation rate and strength)
    feature_scores = []
    for k in range(ibp_model.n_features):
        score = (feature_interpretations[k]['activation_rate'] * 
                feature_interpretations[k]['feature_strength'])
        feature_scores.append((k, score))
    
    # Get top features
    top_feature_indices = [k for k, _ in sorted(feature_scores, key=lambda x: x[1], 
                                              reverse=True)[:top_features]]
    
    # Calculate co-occurrence matrix
    Z_subset = ibp_model.Z[:, top_feature_indices]
    cooccurrence = np.dot(Z_subset.T, Z_subset)
    
    # Create network
    G = nx.Graph()
    
    # Add nodes
    for i, k in enumerate(top_feature_indices):
        top_terms = [term for term, _ in feature_interpretations[k]['positive_terms'][:3]]
        label = f"F{k}: {', '.join(top_terms)}"
        G.add_node(i, label=label, feature_id=k)
    
    # Add edges for significant co-occurrences
    threshold = np.percentile(cooccurrence[cooccurrence > 0], 75)
    
    for i in range(len(top_feature_indices)):
        for j in range(i+1, len(top_feature_indices)):
            if cooccurrence[i, j] > threshold:
                weight = cooccurrence[i, j]
                G.add_edge(i, j, weight=weight)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    node_sizes = [feature_interpretations[top_feature_indices[i]]['n_active_docs'] * 10 
                 for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                          alpha=0.7)
    
    # Draw edges
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=[w/max(edge_weights)*5 for w in edge_weights], 
                          alpha=0.6)
    
    # Draw labels
    labels = {i: f"F{top_feature_indices[i]}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title('IBP Feature Co-occurrence Network\n(Node size = # active documents, Edge width = co-occurrence strength)', 
              fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_multi_topic_coverage(ibp_model, metadata):
    """Analyze how well IBP captures multi-topic documents"""
    
    analysis = {
        'single_topic': {'features_per_doc': [], 'doc_indices': []},
        'multi_topic': {'features_per_doc': [], 'doc_indices': []}
    }
    
    features_per_doc = np.sum(ibp_model.Z, axis=1)
    
    for i, meta in enumerate(metadata):
        topic_cat = meta['topic_category']
        analysis[topic_cat]['features_per_doc'].append(features_per_doc[i])
        analysis[topic_cat]['doc_indices'].append(i)
    
    # Statistical comparison
    single_features = analysis['single_topic']['features_per_doc']
    multi_features = analysis['multi_topic']['features_per_doc']
    
    print("\n=== MULTI-TOPIC ANALYSIS ===")
    print(f"Single-topic docs: {len(single_features)} documents")
    print(f"  Average features per doc: {np.mean(single_features):.2f} ± {np.std(single_features):.2f}")
    print(f"  Median features per doc: {np.median(single_features):.2f}")
    
    print(f"\nMulti-topic docs: {len(multi_features)} documents")  
    print(f"  Average features per doc: {np.mean(multi_features):.2f} ± {np.std(multi_features):.2f}")
    print(f"  Median features per doc: {np.median(multi_features):.2f}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist([single_features, multi_features], bins=15, alpha=0.7, 
             label=['Single-topic', 'Multi-topic'], color=['blue', 'orange'])
    plt.xlabel('Number of Features per Document')
    plt.ylabel('Number of Documents')
    plt.title('Feature Distribution by Document Type')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.boxplot([single_features, multi_features], labels=['Single-topic', 'Multi-topic'])
    plt.ylabel('Number of Features per Document')
    plt.title('Feature Count Distribution')
    
    plt.tight_layout()
    plt.show()
    
def save_results_to_files(results, output_dir="ibp_wikipedia_results"):
    """
    Save comprehensive IBP results to multiple file formats
    
    Args:
        results: Dictionary returned from main() function
        output_dir: Directory to save results
    """
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"\n=== SAVING RESULTS TO {full_output_dir} ===")
    
    # 1. Save main results dataframe
    results_file = os.path.join(full_output_dir, "document_analysis.csv")
    results['results_dataframe'].to_csv(results_file, index=False)
    print(f"✓ Saved document analysis to: {results_file}")
    
    # 2. Save feature matrix (binary activations)
    feature_matrix_file = os.path.join(full_output_dir, "feature_matrix.csv")
    feature_df = pd.DataFrame(results['feature_matrix'], 
                             columns=[f"feature_{i}" for i in range(results['feature_matrix'].shape[1])])
    feature_df.insert(0, 'doc_id', range(len(feature_df)))
    feature_df.insert(1, 'title', [m['title'] for m in results['metadata']])
    feature_df.to_csv(feature_matrix_file, index=False)
    print(f"✓ Saved feature matrix to: {feature_matrix_file}")
    
    # 3. Save feature interpretations
    feature_interp_file = os.path.join(full_output_dir, "feature_interpretations.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_interpretations = {}
    for k, interp in results['feature_interpretations'].items():
        serializable_interpretations[str(k)] = {
            'positive_terms': interp['positive_terms'],
            'negative_terms': interp['negative_terms'],
            'activation_rate': float(interp['activation_rate']),
            'n_active_docs': int(interp['n_active_docs']),
            'feature_strength': float(interp['feature_strength'])
        }
    
    with open(feature_interp_file, 'w') as f:
        json.dump(serializable_interpretations, f, indent=2)
    print(f"✓ Saved feature interpretations to: {feature_interp_file}")
    
    # 4. Save detailed feature analysis
    feature_analysis_file = os.path.join(full_output_dir, "detailed_feature_analysis.txt")
    with open(feature_analysis_file, 'w', encoding='utf-8') as f:
        f.write("=== IBP WIKIPEDIA FEATURE ANALYSIS ===\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Documents: {len(results['documents'])}\n")
        f.write(f"Total Features: {results['ibp_model'].n_features}\n")
        f.write(f"Feature Matrix Sparsity: {1 - np.mean(results['feature_matrix']):.3f}\n\n")
        
        f.write("=== FEATURE DETAILS ===\n\n")
        for k in sorted(results['feature_interpretations'].keys()):
            interp = results['feature_interpretations'][k]
            f.write(f"Feature {k}:\n")
            f.write(f"  Active in {interp['n_active_docs']} documents ({interp['activation_rate']:.1%})\n")
            f.write(f"  Feature strength: {interp['feature_strength']:.4f}\n")
            
            f.write("  Positive indicators: ")
            pos_terms = [f"{term}({weight:.3f})" for term, weight in interp['positive_terms'][:8]]
            f.write(", ".join(pos_terms) + "\n")
            
            if interp['negative_terms']:
                f.write("  Negative indicators: ")
                neg_terms = [f"{term}({weight:.3f})" for term, weight in interp['negative_terms'][:5]]
                f.write(", ".join(neg_terms) + "\n")
            f.write("\n")
        
        # Add bridge documents analysis
        f.write("=== BRIDGE DOCUMENTS (Multi-Featured) ===\n\n")
        for title, n_feat, category in results['bridge_documents'][:20]:
            f.write(f"• {title} ({n_feat} features, {category})\n")
        
        # Add feature co-activations
        f.write("\n=== TOP FEATURE CO-ACTIVATIONS ===\n\n")
        for f1, f2, coact, terms1, terms2 in results['feature_pairs'][:15]:
            f.write(f"F{f1}({','.join(terms1)}) ↔ F{f2}({','.join(terms2)}): {coact} documents\n")
    
    print(f"✓ Saved detailed analysis to: {feature_analysis_file}")
    
    # 5. Save multi-topic analysis
    multi_topic_file = os.path.join(full_output_dir, "multi_topic_analysis.json")
    serializable_analysis = {}
    for category, data in results['multi_topic_analysis'].items():
        serializable_analysis[category] = {
            'features_per_doc': data['features_per_doc'],
            'doc_indices': data['doc_indices'],
            'mean_features': float(np.mean(data['features_per_doc'])),
            'std_features': float(np.std(data['features_per_doc'])),
            'median_features': float(np.median(data['features_per_doc']))
        }
    
    with open(multi_topic_file, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    print(f"✓ Saved multi-topic analysis to: {multi_topic_file}")
    
    # 6. Save raw documents with metadata
    documents_file = os.path.join(full_output_dir, "raw_documents.json")
    documents_data = []
    for i, (doc, meta) in enumerate(zip(results['documents'], results['metadata'])):
        documents_data.append({
            'doc_id': i,
            'title': meta['title'],
            'topic_category': meta['topic_category'],
            'search_query': meta['search_query'],
            'content': doc,
            'active_features': results['feature_matrix'][i].tolist(),
            'n_features': int(np.sum(results['feature_matrix'][i]))
        })
    
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(documents_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved raw documents to: {documents_file}")
    
    # 7. Save model objects (pickle)
    models_file = os.path.join(full_output_dir, "models.pkl")
    model_objects = {
        'ibp_model': results['ibp_model'],
        'vectorizer': results['vectorizer']
    }
    
    with open(models_file, 'wb') as f:
        pickle.dump(model_objects, f)
    print(f"✓ Saved model objects to: {models_file}")
    
    # 8. Save summary statistics
    summary_file = os.path.join(full_output_dir, "summary_statistics.json")
    
    features_per_doc = np.sum(results['feature_matrix'], axis=1)
    docs_per_feature = np.sum(results['feature_matrix'], axis=0)
    
    summary_stats = {
        'experiment_info': {
            'timestamp': timestamp,
            'total_documents': len(results['documents']),
            'total_features': int(results['ibp_model'].n_features),
            'alpha_parameter': float(results['ibp_model'].alpha),
            'sigma_x': float(results['ibp_model'].sigma_x),
            'sigma_a': float(results['ibp_model'].sigma_a)
        },
        'document_statistics': {
            'single_topic_docs': len(results['multi_topic_analysis']['single_topic']['doc_indices']),
            'multi_topic_docs': len(results['multi_topic_analysis']['multi_topic']['doc_indices']),
            'avg_features_per_doc': float(np.mean(features_per_doc)),
            'std_features_per_doc': float(np.std(features_per_doc)),
            'median_features_per_doc': float(np.median(features_per_doc)),
            'max_features_per_doc': int(np.max(features_per_doc)),
            'min_features_per_doc': int(np.min(features_per_doc))
        },
        'feature_statistics': {
            'avg_docs_per_feature': float(np.mean(docs_per_feature)),
            'std_docs_per_feature': float(np.std(docs_per_feature)),
            'median_docs_per_feature': float(np.median(docs_per_feature)),
            'most_active_feature_docs': int(np.max(docs_per_feature)),
            'least_active_feature_docs': int(np.min(docs_per_feature)),
            'sparsity': float(1 - np.mean(results['feature_matrix']))
        },
        'complexity_distribution': {
            'simple_docs_1_2_features': int(np.sum((features_per_doc >= 1) & (features_per_doc <= 2))),
            'moderate_docs_3_4_features': int(np.sum((features_per_doc >= 3) & (features_per_doc <= 4))),
            'complex_docs_5plus_features': int(np.sum(features_per_doc >= 5))
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✓ Saved summary statistics to: {summary_file}")
    
    # 9. Create a README file
    readme_file = os.path.join(full_output_dir, "README.md")
    with open(readme_file, 'w') as f:
        f.write("# IBP Wikipedia Document Analysis Results\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Documents:** {len(results['documents'])}\n")
        f.write(f"**Latent Features Discovered:** {results['ibp_model'].n_features}\n\n")
        
        f.write("## File Descriptions\n\n")
        f.write("- `document_analysis.csv`: Main results with document titles, categories, and feature counts\n")
        f.write("- `feature_matrix.csv`: Binary matrix showing which features are active for each document\n")
        f.write("- `feature_interpretations.json`: Detailed interpretation of each latent feature\n")
        f.write("- `detailed_feature_analysis.txt`: Human-readable comprehensive analysis\n")
        f.write("- `multi_topic_analysis.json`: Statistical comparison of single vs multi-topic documents\n")
        f.write("- `raw_documents.json`: Full document text with metadata and feature activations\n")
        f.write("- `models.pkl`: Pickled IBP model and vectorizer for reuse\n")
        f.write("- `summary_statistics.json`: Key statistics and model parameters\n\n")
        
        f.write("## Key Findings\n\n")
        single_avg = np.mean(results['multi_topic_analysis']['single_topic']['features_per_doc'])
        multi_avg = np.mean(results['multi_topic_analysis']['multi_topic']['features_per_doc'])
        f.write(f"- Single-topic documents average {single_avg:.2f} features\n")
        f.write(f"- Multi-topic documents average {multi_avg:.2f} features\n")
        f.write(f"- Multi-topic documents show {multi_avg/single_avg:.2f}x more feature complexity\n")
        f.write(f"- Feature matrix sparsity: {1 - np.mean(results['feature_matrix']):.3f}\n")
        f.write(f"- Found {len(results['bridge_documents'])} potential bridge documents\n")
        f.write(f"- Identified {len(results['feature_pairs'])} significant feature co-activations\n")
    
    print(f"✓ Saved README to: {readme_file}")
    
    print(f"\n🎉 All results saved successfully to: {full_output_dir}")
    print(f"📊 Total files created: 9")
    
    return full_output_dir

def main():
    """Main execution function for IBP Wikipedia analysis"""
    print("=== Indian Buffet Process Wikipedia Document Analysis ===")
    
    # Scrape diverse Wikipedia documents
    documents, metadata = scrape_multi_topic_documents(3500)
    
    print(f"\nSuccessfully scraped {len(documents)} Wikipedia articles")
    print(f"Sample article: {documents[0][:200]}...")
    
    # Analyze document categories
    category_counts = Counter([m['topic_category'] for m in metadata])
    print(f"\nDocument category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} articles")
    
    # Analyze search queries
    query_counts = Counter([m['search_query'] for m in metadata])
    print(f"\nTop search queries:")
    for query, count in query_counts.most_common(10):
        print(f"  {query}: {count} articles")
    
    # Vectorize documents with enhanced features
    print("\nVectorizing documents with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=500,  # More features for richer representation
        stop_words='english',
        max_df=0.8,
        min_df=5,
        ngram_range=(1, 3),  # Include trigrams for better topic capture
        sublinear_tf=True   # Dampen tf scores
    )
    X = vectorizer.fit_transform(documents)
    print(f"Document matrix shape: {X.shape}")
    
    # Run Indian Buffet Process
    print("\nRunning Indian Buffet Process...")
    ibp = IndianBuffetProcess(alpha=3.0, sigma_x=0.5, sigma_a=1.0)
    ibp.fit(X, n_iterations=250, verbose=True)
    
    print(f"\nIBP discovered {ibp.n_features} latent features")
    
    # Get feature activations
    Z = ibp.get_feature_activations()
    features_per_doc = np.sum(Z, axis=1)
    docs_per_feature = np.sum(Z, axis=0)
    
    print(f"Average features per document: {np.mean(features_per_doc):.2f}")
    print(f"Average documents per feature: {np.mean(docs_per_feature):.2f}")
    print(f"Feature activation sparsity: {1 - np.mean(Z):.3f}")
    
    # Extract feature interpretations
    print("\nExtracting feature interpretations...")
    feature_interpretations = extract_feature_interpretations(ibp, vectorizer, top_k=12)
    
    # Display feature analysis
    print("\n=== DISCOVERED LATENT FEATURES ===")
    for k in sorted(feature_interpretations.keys()):
        interp = feature_interpretations[k]
        print(f"\nFeature {k} (Active in {interp['n_active_docs']} docs, "
              f"{interp['activation_rate']:.1%} activation rate):")
        
        print("  Positive indicators:", end=" ")
        pos_terms = [f"{term}({weight:.3f})" for term, weight in interp['positive_terms'][:5]]
        print(", ".join(pos_terms))
        
        if interp['negative_terms']:
            print("  Negative indicators:", end=" ")
            neg_terms = [f"{term}({weight:.3f})" for term, weight in interp['negative_terms'][:3]]
            print(", ".join(neg_terms))
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'doc_id': range(len(documents)),
        'title': [m['title'] for m in metadata],
        'topic_category': [m['topic_category'] for m in metadata],
        'search_query': [m['search_query'] for m in metadata],
        'n_features': features_per_doc,
        'feature_list': [np.where(Z[i])[0].tolist() for i in range(len(documents))],
        'document_preview': [doc[:150] + "..." if len(doc) > 150 else doc for doc in documents]
    })
    
    # Sort by number of features (most multi-featured first)
    results_df = results_df.sort_values('n_features', ascending=False)
    
    print("\n=== MOST MULTI-FEATURED DOCUMENTS ===")
    for _, row in results_df.head(15).iterrows():
        feature_desc = []
        for feat_id in row['feature_list'][:5]:  # Show top 5 features
            top_terms = [term for term, _ in feature_interpretations[feat_id]['positive_terms'][:2]]
            feature_desc.append(f"F{feat_id}({','.join(top_terms)})")
        
        print(f"\n• {row['title']} [{row['topic_category']}]")
        print(f"  Features ({row['n_features']}): {'; '.join(feature_desc)}")
        print(f"  Query: {row['search_query']}")
    
    # Analyze multi-topic coverage
    multi_topic_analysis = analyze_multi_topic_coverage(ibp, metadata)
    
    # Create visualizations
    print("\nCreating IBP visualizations...")
    create_ibp_visualizations(ibp, metadata, feature_interpretations, documents)
    
    # Feature-document analysis
    print("\n=== FEATURE-DOCUMENT INTERSECTION ANALYSIS ===")
    
    # Find documents that bridge multiple topic areas
    bridge_documents = []
    for i, row in results_df.iterrows():
        if row['n_features'] >= 4:  # Documents with many features
            # Check if features span different semantic areas
            feature_terms = []
            for feat_id in row['feature_list']:
                terms = [term for term, _ in feature_interpretations[feat_id]['positive_terms'][:2]]
                feature_terms.extend(terms)
            
            # Simple heuristic: if document has diverse terms, it's likely bridging topics
            unique_stems = set()
            for term in feature_terms:
                # Simple stemming approximation
                stem = term[:4] if len(term) > 4 else term
                unique_stems.add(stem)
            
            if len(unique_stems) >= row['n_features']:  # Diverse feature vocabulary
                bridge_documents.append((row['title'], row['n_features'], row['topic_category']))
    
    print(f"Found {len(bridge_documents)} potential bridge documents:")
    for title, n_feat, category in bridge_documents[:10]:
        print(f"  • {title} ({n_feat} features, {category})")
    
    # Feature co-activation analysis
    print("\n=== FEATURE CO-ACTIVATION PATTERNS ===")
    coactivation_matrix = np.dot(Z.T, Z)
    np.fill_diagonal(coactivation_matrix, 0)  # Remove self-activation
    
    # Find strongest feature pairs
    feature_pairs = []
    for i in range(ibp.n_features):
        for j in range(i+1, ibp.n_features):
            coactivation = coactivation_matrix[i, j]
            if coactivation >= 5:  # Threshold for significant co-activation
                terms_i = [term for term, _ in feature_interpretations[i]['positive_terms'][:2]]
                terms_j = [term for term, _ in feature_interpretations[j]['positive_terms'][:2]]
                feature_pairs.append((i, j, coactivation, terms_i, terms_j))
    
    # Sort by co-activation strength
    feature_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("Strongest feature co-activations:")
    for f1, f2, coact, terms1, terms2 in feature_pairs[:10]:
        print(f"  F{f1}({','.join(terms1)}) ↔ F{f2}({','.join(terms2)}): {coact} docs")
    
    # Topic category analysis by features
    print("\n=== FEATURE PREFERENCE BY DOCUMENT TYPE ===")
    single_topic_features = Z[np.array([m['topic_category'] for m in metadata]) == 'single']
    multi_topic_features = Z[np.array([m['topic_category'] for m in metadata]) == 'multi']
    
    single_feature_rates = np.mean(single_topic_features, axis=0)
    multi_feature_rates = np.mean(multi_topic_features, axis=0)
    
    # Find features that prefer multi-topic documents
    multi_preferring_features = []
    for k in range(ibp.n_features):
        if multi_feature_rates[k] > single_feature_rates[k] * 1.5:  # 50% higher in multi-topic
            preference_ratio = multi_feature_rates[k] / (single_feature_rates[k] + 1e-6)
            top_terms = [term for term, _ in feature_interpretations[k]['positive_terms'][:3]]
            multi_preferring_features.append((k, preference_ratio, top_terms))
    
    multi_preferring_features.sort(key=lambda x: x[1], reverse=True)
    
    print("Features preferentially active in multi-topic documents:")
    for feat_id, ratio, terms in multi_preferring_features[:8]:
        print(f"  F{feat_id}: {','.join(terms)} (ratio: {ratio:.2f})")
    
    # Summary statistics
    print("\n=== FINAL SUMMARY ===")
    print(f"Total Wikipedia articles analyzed: {len(documents)}")
    print(f"Latent features discovered: {ibp.n_features}")
    print(f"Average features per document: {np.mean(features_per_doc):.2f} ± {np.std(features_per_doc):.2f}")
    print(f"Feature activation sparsity: {1 - np.mean(Z):.3f}")
    print(f"Multi-topic documents show {np.mean(multi_topic_analysis['multi_topic']['features_per_doc']) / np.mean(multi_topic_analysis['single_topic']['features_per_doc']):.2f}x more features on average")
    
    # Document complexity distribution
    complexity_bins = {
        'Simple (1-2 features)': np.sum((features_per_doc >= 1) & (features_per_doc <= 2)),
        'Moderate (3-4 features)': np.sum((features_per_doc >= 3) & (features_per_doc <= 4)),
        'Complex (5+ features)': np.sum(features_per_doc >= 5)
    }
    
    print(f"\nDocument complexity distribution:")
    for complexity, count in complexity_bins.items():
        print(f"  {complexity}: {count} documents ({count/len(documents):.1%})")
    
    # Save all results to files
    print("\n" + "="*60)
    output_directory = save_results_to_files({
        'documents': documents,
        'metadata': metadata,
        'ibp_model': ibp,
        'feature_matrix': Z,
        'feature_interpretations': feature_interpretations,
        'results_dataframe': results_df,
        'vectorizer': vectorizer,
        'multi_topic_analysis': multi_topic_analysis,
        'bridge_documents': bridge_documents,
        'feature_pairs': feature_pairs
    })
    
    return {
        'documents': documents,
        'metadata': metadata,
        'ibp_model': ibp,
        'feature_matrix': Z,
        'feature_interpretations': feature_interpretations,
        'results_dataframe': results_df,
        'vectorizer': vectorizer,
        'multi_topic_analysis': multi_topic_analysis,
        'bridge_documents': bridge_documents,
        'feature_pairs': feature_pairs,
        'output_directory': output_directory
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the full IBP pipeline
    results = main()