import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
import random
import re
from scipy.spatial.distance import cosine
from scipy.stats import multivariate_normal
import seaborn as sns
import time
import requests
from bs4 import BeautifulSoup
import urllib.parse

class ChineseRestaurantProcess:
    def __init__(self, alpha=1.0, base_distribution_params=None):
        """
        Chinese Restaurant Process for document clustering
        
        Args:
            alpha: Concentration parameter (higher = more clusters)
            base_distribution_params: Parameters for base distribution
        """
        self.alpha = alpha
        self.clusters = {}
        self.cluster_assignments = []
        self.cluster_params = {}
        self.n_clusters = 0
        
    def fit(self, X, n_iterations=100):
        """
        Fit CRP model to document vectors
        
        Args:
            X: Document-term matrix (n_docs x n_features)
            n_iterations: Number of Gibbs sampling iterations
        """
        n_docs, n_features = X.shape
        
        # Initialize: assign each document to its own cluster
        self.cluster_assignments = list(range(n_docs))
        self.n_clusters = n_docs
        
        # Initialize cluster parameters (mean vectors)
        self.cluster_params = {}
        for i in range(n_docs):
            if hasattr(X, 'toarray'):
                self.cluster_params[i] = X[i].toarray().flatten()
            else:
                self.cluster_params[i] = X[i].copy()
        
        # Gibbs sampling
        for iteration in range(n_iterations):
            for doc_idx in range(n_docs):
                # Remove document from current cluster
                old_cluster = self.cluster_assignments[doc_idx]
                self._remove_document_from_cluster(doc_idx, old_cluster, X)
                
                # Calculate probabilities for each existing cluster + new cluster
                probs = self._calculate_cluster_probabilities(doc_idx, X)
                
                # Sample new cluster assignment
                new_cluster = self._sample_cluster(probs)
                
                # Assign document to new cluster
                self._assign_document_to_cluster(doc_idx, new_cluster, X)
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: {len(set(self.cluster_assignments))} clusters")
        
        # Clean up empty clusters and renumber
        self._cleanup_clusters()
        
    def _remove_document_from_cluster(self, doc_idx, cluster_id, X):
        """Remove document from cluster and update cluster parameters"""
        # Count documents in this cluster
        cluster_docs = [i for i, c in enumerate(self.cluster_assignments) if c == cluster_id]
        
        if len(cluster_docs) == 1:  # Only this document in cluster
            # Remove empty cluster
            if cluster_id in self.cluster_params:
                del self.cluster_params[cluster_id]
        else:
            # Update cluster mean (remove this document's contribution)
            cluster_docs_except_current = [i for i in cluster_docs if i != doc_idx]
            if cluster_docs_except_current:
                if hasattr(X, 'toarray'):
                    cluster_vectors = X[cluster_docs_except_current].toarray()
                else:
                    cluster_vectors = X[cluster_docs_except_current]
                self.cluster_params[cluster_id] = np.mean(cluster_vectors, axis=0)
    
    def _assign_document_to_cluster(self, doc_idx, cluster_id, X):
        """Assign document to cluster and update parameters"""
        self.cluster_assignments[doc_idx] = cluster_id
        
        # Update cluster parameters
        cluster_docs = [i for i, c in enumerate(self.cluster_assignments) if c == cluster_id]
        if hasattr(X, 'toarray'):
            cluster_vectors = X[cluster_docs].toarray()
        else:
            cluster_vectors = X[cluster_docs]
        self.cluster_params[cluster_id] = np.mean(cluster_vectors, axis=0)
    
    def _calculate_cluster_probabilities(self, doc_idx, X):
        """Calculate probability of assigning document to each cluster"""
        doc_vector = X[doc_idx].toarray().flatten() if hasattr(X, 'toarray') else X[doc_idx]
        existing_clusters = list(set(self.cluster_assignments))
        probs = []
        
        # Probability for existing clusters
        for cluster_id in existing_clusters:
            cluster_size = self.cluster_assignments.count(cluster_id)
            
            # CRP probability (proportional to cluster size)
            crp_prob = cluster_size / (len(self.cluster_assignments) + self.alpha - 1)
            
            # Likelihood (similarity to cluster center)
            if cluster_id in self.cluster_params:
                cluster_center = self.cluster_params[cluster_id]
                # Ensure both vectors are dense
                if hasattr(cluster_center, 'toarray'):
                    cluster_center = cluster_center.flatten()
                
                # Calculate cosine similarity manually to avoid issues
                dot_product = np.dot(doc_vector, cluster_center)
                norm_doc = np.linalg.norm(doc_vector)
                norm_center = np.linalg.norm(cluster_center)
                
                if norm_doc > 0 and norm_center > 0:
                    similarity = dot_product / (norm_doc * norm_center)
                    similarity = max(similarity, 0.01)  # Avoid zero likelihood
                else:
                    similarity = 0.01
                
                likelihood = similarity
            else:
                likelihood = 0.01
            
            probs.append((cluster_id, crp_prob * likelihood))
        
        # Probability for new cluster
        new_cluster_id = max(existing_clusters) + 1 if existing_clusters else 0
        crp_prob = self.alpha / (len(self.cluster_assignments) + self.alpha - 1)
        probs.append((new_cluster_id, crp_prob * 0.5))  # Higher base probability for new cluster
        
        return probs
    
    def _sample_cluster(self, cluster_probs):
        """Sample cluster assignment based on probabilities"""
        clusters, probs = zip(*cluster_probs)
        probs = np.array(probs)
        probs = probs / np.sum(probs)  # Normalize
        
        return np.random.choice(clusters, p=probs)
    
    def _cleanup_clusters(self):
        """Remove empty clusters and renumber consecutively"""
        unique_clusters = sorted(list(set(self.cluster_assignments)))
        cluster_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        
        self.cluster_assignments = [cluster_mapping[c] for c in self.cluster_assignments]
        
        # Update cluster parameters
        new_params = {}
        for old_id, new_id in cluster_mapping.items():
            if old_id in self.cluster_params:
                new_params[new_id] = self.cluster_params[old_id]
        self.cluster_params = new_params
        
        self.n_clusters = len(unique_clusters)

class WikipediaScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CRP-Research-Tool/1.0 (Educational Research Purpose)'
        })
        self.base_url = "https://en.wikipedia.org"
        self.api_url = "https://en.wikipedia.org/api/rest_v1"
        
    def search_articles(self, query, limit=50):
        """Search for Wikipedia articles by query"""
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
                    articles.append({
                        'title': result['title'],
                        'pageid': result['pageid'],
                        'size': result.get('size', 0),
                        'wordcount': result.get('wordcount', 0)
                    })
            
            return articles
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    def get_featured_articles(self, limit=300):
        """Get Wikipedia featured articles"""
        featured_url = f"{self.base_url}/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'categorymembers',
            'cmtitle': 'Category:Featured articles',
            'cmlimit': limit,
            'cmtype': 'page'
        }
        
        try:
            response = self.session.get(featured_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            if 'query' in data and 'categorymembers' in data['query']:
                for result in data['query']['categorymembers']:
                    articles.append({
                        'title': result['title'],
                        'pageid': result['pageid']
                    })
            
            return articles
        except Exception as e:
            print(f"Error getting featured articles: {e}")
            return []
    
    def get_article_content(self, title):
        """Get clean text content from a Wikipedia article"""
        try:
            # Get page content via API
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
                    
                    # Clean the text
                    text = self.clean_text(text)
                    
                    # Return first reasonable chunk (avoid huge articles)
                    if len(text) > 1000:
                        # Split into paragraphs and take first few
                        paragraphs = text.split('\n\n')
                        selected_paragraphs = []
                        char_count = 0
                        
                        for para in paragraphs:
                            if char_count + len(para) > 2000:  # Limit article length
                                break
                            selected_paragraphs.append(para)
                            char_count += len(para)
                        
                        return '\n\n'.join(selected_paragraphs)
                    else:
                        return text
            
            return None
        except Exception as e:
            print(f"Error getting content for '{title}': {e}")
            return None
    
    def clean_text(self, text):
        """Clean Wikipedia text"""
        if not text:
            return ""
        
        # Remove common Wikipedia artifacts
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[.*?\]', '', text)  # Remove references
        text = re.sub(r'\n+', '\n\n', text)  # Normalize newlines
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        return text.strip()

def scrape_wikipedia_documents(target_count=2000):
    """Scrape Wikipedia articles from featured articles and specific topics"""
    
    scraper = WikipediaScraper()
    
    # Define topics to search
    topics = [
        "cricket sport",
        "basketball sport", 
        "machine learning",
        "number theory mathematics",
        "climate change",
        "genomics biology",
        "Indian politics",
        "Indian classical music",
        "opera music",
        "bebop jazz music",
        "trees botany",
        "mountains geography"
    ]
    
    all_articles = []
    scraped_titles = set()  # Avoid duplicates
    
    print("Scraping Wikipedia articles...")
    
    # Get featured articles
    print("Fetching featured articles...")
    featured = scraper.get_featured_articles(300)
    
    for article in featured[:200]:  # Limit featured articles
        if len(all_articles) >= target_count:
            break
            
        title = article['title']
        if title in scraped_titles:
            continue
            
        content = scraper.get_article_content(title)
        if content and len(content) > 200:  # Minimum content length
            all_articles.append({
                'title': title,
                'content': content,
                'topic': 'featured',
                'source': 'wikipedia'
            })
            scraped_titles.add(title)
            
            if len(all_articles) % 50 == 0:
                print(f"Scraped {len(all_articles)} articles so far...")
        
        time.sleep(0.1)  # Rate limiting
    
    print(f"Scraped {len(all_articles)} featured articles")
    
    # Get topic-specific articles
    articles_per_topic = (target_count - len(all_articles)) // len(topics)
    
    for topic in topics:
        if len(all_articles) >= target_count:
            break
            
        print(f"Scraping articles for: {topic}")
        
        # Search for articles on this topic
        search_results = scraper.search_articles(topic, limit=articles_per_topic + 20)
        
        topic_count = 0
        for article in search_results:
            if len(all_articles) >= target_count or topic_count >= articles_per_topic:
                break
                
            title = article['title']
            if title in scraped_titles:
                continue
                
            content = scraper.get_article_content(title)
            if content and len(content) > 200:
                all_articles.append({
                    'title': title,
                    'content': content,
                    'topic': topic,
                    'source': 'wikipedia'
                })
                scraped_titles.add(title)
                topic_count += 1
            
            time.sleep(0.1)  # Rate limiting
        
        print(f"Scraped {topic_count} articles for {topic}")
    
    print(f"\nTotal articles scraped: {len(all_articles)}")
    
    # Extract just the content and metadata
    documents = [article['content'] for article in all_articles]
    metadata = [{'title': article['title'], 'topic': article['topic']} for article in all_articles]
    
    return documents, metadata

def extract_cluster_keywords(documents, cluster_assignments, vectorizer, top_k=10):
    """Extract top keywords for each cluster with weights"""
    
    cluster_keywords = {}
    unique_clusters = sorted(list(set(cluster_assignments)))
    
    for cluster_id in unique_clusters:
        # Get documents in this cluster
        cluster_docs = [documents[i] for i, c in enumerate(cluster_assignments) if c == cluster_id]
        
        if not cluster_docs:
            continue
        
        # Create TF-IDF for this cluster vs all other documents
        all_other_docs = [documents[i] for i, c in enumerate(cluster_assignments) if c != cluster_id]
        
        if not all_other_docs:
            all_other_docs = cluster_docs  # Fallback if only one cluster
        
        # Combine cluster documents
        cluster_text = ' '.join(cluster_docs)
        other_text = ' '.join(all_other_docs)
        
        # Calculate TF-IDF
        temp_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = temp_vectorizer.fit_transform([cluster_text, other_text])
        
        feature_names = temp_vectorizer.get_feature_names_out()
        cluster_scores = tfidf_matrix[0].toarray()[0]
        
        # Get top keywords
        top_indices = cluster_scores.argsort()[-top_k:][::-1]
        keywords = [(feature_names[i], cluster_scores[i]) for i in top_indices if cluster_scores[i] > 0]
        
        cluster_keywords[cluster_id] = keywords
    
    return cluster_keywords

def plot_pca_clusters(X, cluster_assignments, cluster_keywords, metadata, title="CRP Wikipedia Document Clustering"):
    """Create PCA visualization with cluster overlays"""
    
    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
    
    # Create the plot
    plt.figure(figsize=(16, 12))
    
    # Color palette
    unique_clusters = sorted(list(set(cluster_assignments)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = np.array(cluster_assignments) == cluster_id
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=40)
    
    # Add cluster centers and labels
    for cluster_id in unique_clusters:
        cluster_mask = np.array(cluster_assignments) == cluster_id
        if np.sum(cluster_mask) > 0:
            center_x = np.mean(X_pca[cluster_mask, 0])
            center_y = np.mean(X_pca[cluster_mask, 1])
            plt.scatter(center_x, center_y, c='black', marker='x', s=300, linewidth=4)
            
            # Add top keywords as text
            if cluster_id in cluster_keywords and cluster_keywords[cluster_id]:
                top_words = [word for word, _ in cluster_keywords[cluster_id][:3]]
                plt.annotate(', '.join(top_words), (center_x, center_y), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    plt.title(f'{title}\n({len(unique_clusters)} clusters found)', fontsize=16, weight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return X_pca, pca

def analyze_cluster_topics(cluster_assignments, metadata):
    """Analyze which Wikipedia topics ended up in which clusters"""
    
    cluster_topic_analysis = {}
    unique_clusters = sorted(list(set(cluster_assignments)))
    
    for cluster_id in unique_clusters:
        cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
        cluster_topics = [metadata[i]['topic'] for i in cluster_indices]
        cluster_titles = [metadata[i]['title'] for i in cluster_indices]
        
        topic_counts = Counter(cluster_topics)
        
        cluster_topic_analysis[cluster_id] = {
            'size': len(cluster_indices),
            'topic_distribution': dict(topic_counts),
            'sample_titles': cluster_titles[:5]  # First 5 titles as examples
        }
    
    return cluster_topic_analysis

def main():
    """Main execution function"""
    print("=== Wikipedia CRP Document Clustering ===")
    
    # Scrape Wikipedia documents
    documents, metadata = scrape_wikipedia_documents(2000)
    
    print(f"\nSuccessfully scraped {len(documents)} Wikipedia articles")
    print(f"Sample article: {documents[0][:150]}...")
    
    # Analyze scraped topics
    topic_counts = Counter([m['topic'] for m in metadata])
    print(f"\nTopic distribution:")
    for topic, count in topic_counts.most_common():
        print(f"  {topic}: {count} articles")
    
    # Vectorize documents
    print("\nVectorizing documents with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=300, 
        stop_words='english', 
        max_df=0.7, 
        min_df=3, 
        ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(documents)
    print(f"Document matrix shape: {X.shape}")
    
    # Run Chinese Restaurant Process
    print("\nRunning Chinese Restaurant Process...")
    crp = ChineseRestaurantProcess(alpha=4.0)  # Higher alpha for more clusters
    crp.fit(X, n_iterations=150)
    
    print(f"\nCRP found {crp.n_clusters} clusters")
    cluster_sizes = Counter(crp.cluster_assignments)
    print(f"Cluster sizes: {dict(cluster_sizes)}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'doc_id': range(len(documents)),
        'cluster_id': crp.cluster_assignments,
        'title': [m['title'] for m in metadata],
        'topic': [m['topic'] for m in metadata],
        'document_text': [doc[:200] + "..." if len(doc) > 200 else doc for doc in documents]
    })
    
    # Sort by cluster ID
    results_df = results_df.sort_values('cluster_id')
    
    print("\n=== DOCUMENT ASSIGNMENTS (sorted by cluster ID) ===")
    for cluster_id in sorted(results_df['cluster_id'].unique()):
        cluster_docs = results_df[results_df['cluster_id'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
        for _, row in cluster_docs.head(5).iterrows():  # Show first 5 docs per cluster
            print(f"  • {row['title']} [{row['topic']}]")
    
    # Extract cluster keywords
    print("\nExtracting cluster keywords...")
    cluster_keywords = extract_cluster_keywords(documents, crp.cluster_assignments, vectorizer, top_k=8)
    
    print("\n=== CLUSTER KEYWORDS WITH WEIGHTS ===")
    for cluster_id in sorted(cluster_keywords.keys()):
        print(f"\nCluster {cluster_id}:")
        for word, weight in cluster_keywords[cluster_id]:
            print(f"  {word}: {weight:.4f}")
    
    # Analyze cluster composition by original topics
    print("\n=== CLUSTER-TOPIC ANALYSIS ===")
    cluster_analysis = analyze_cluster_topics(crp.cluster_assignments, metadata)
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\nCluster {cluster_id} ({analysis['size']} articles):")
        print(f"  Topic distribution: {analysis['topic_distribution']}")
        print(f"  Sample articles: {', '.join(analysis['sample_titles'][:3])}")
    
    # Create PCA visualization
    print("\nCreating PCA visualization...")
    X_pca, pca_model = plot_pca_clusters(X, crp.cluster_assignments, cluster_keywords, metadata)
    
    print("\n=== SUMMARY ===")
    print(f"Total Wikipedia articles: {len(documents)}")
    print(f"Number of clusters found: {crp.n_clusters}")
    print(f"Alpha parameter used: {crp.alpha}")
    print(f"PCA explained variance: PC1={pca_model.explained_variance_ratio_[0]:.1%}, PC2={pca_model.explained_variance_ratio_[1]:.1%}")
    print(f"Total variance in 2D: {sum(pca_model.explained_variance_ratio_):.1%}")
    
    return {
        'documents': documents,
        'metadata': metadata,
        'cluster_assignments': crp.cluster_assignments,
        'cluster_keywords': cluster_keywords,
        'cluster_analysis': cluster_analysis,
        'results_dataframe': results_df,
        'pca_coordinates': X_pca,
        'pca_model': pca_model,
        'vectorizer': vectorizer,
        'crp_model': crp
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run the full pipeline
    results = main()
