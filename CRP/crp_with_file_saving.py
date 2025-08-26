import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
import random
import re
from scipy.spatial.distance import cosine
import time
import requests
import json
import pickle
import os
from datetime import datetime

print("All modules loaded")

class ChineseRestaurantProcess:
    def __init__(self, alpha=1.0):
        """
        Chinese Restaurant Process for document clustering
        
        Args:
            alpha: Concentration parameter (higher = more clusters)
        """
        self.alpha = alpha
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
        cluster_docs = [i for i, c in enumerate(self.cluster_assignments) if c == cluster_id]
        
        if len(cluster_docs) == 1:
            # Remove empty cluster
            if cluster_id in self.cluster_params:
                del self.cluster_params[cluster_id]
        else:
            # Update cluster mean
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
                if hasattr(cluster_center, 'toarray'):
                    cluster_center = cluster_center.flatten()
                
                # Calculate cosine similarity
                dot_product = np.dot(doc_vector, cluster_center)
                norm_doc = np.linalg.norm(doc_vector)
                norm_center = np.linalg.norm(cluster_center)
                
                if norm_doc > 0 and norm_center > 0:
                    similarity = dot_product / (norm_doc * norm_center)
                    similarity = max(similarity, 0.01)
                else:
                    similarity = 0.01
                
                likelihood = similarity
            else:
                likelihood = 0.01
            
            probs.append((cluster_id, crp_prob * likelihood))
        
        # Probability for new cluster
        new_cluster_id = max(existing_clusters) + 1 if existing_clusters else 0
        crp_prob = self.alpha / (len(self.cluster_assignments) + self.alpha - 1)
        probs.append((new_cluster_id, crp_prob * 0.5))
        
        return probs
    
    def _sample_cluster(self, cluster_probs):
        """Sample cluster assignment based on probabilities"""
        clusters, probs = zip(*cluster_probs)
        probs = np.array(probs)
        probs = probs / np.sum(probs)
        
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
                    
                    if len(text) > 1000:
                        paragraphs = text.split('\n\n')
                        selected_paragraphs = []
                        char_count = 0
                        
                        for para in paragraphs:
                            if char_count + len(para) > 2000:
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
        
        text = re.sub(r'\[edit\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\n+', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

def scrape_wikipedia_documents(target_count=2000):
    """Scrape Wikipedia articles from featured articles and specific topics"""
    
    scraper = WikipediaScraper()
    
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
    scraped_titles = set()
    
    print("Scraping Wikipedia articles...")
    
    # Get featured articles
    print("Fetching featured articles...")
    featured = scraper.get_featured_articles(300)
    
    for article in featured[:200]:
        if len(all_articles) >= target_count:
            break
            
        title = article['title']
        if title in scraped_titles:
            continue
            
        content = scraper.get_article_content(title)
        if content and len(content) > 200:
            all_articles.append({
                'title': title,
                'content': content,
                'topic': 'featured',
                'source': 'wikipedia'
            })
            scraped_titles.add(title)
            
            if len(all_articles) % 50 == 0:
                print(f"Scraped {len(all_articles)} articles so far...")
        
        time.sleep(0.1)
    
    print(f"Scraped {len(all_articles)} featured articles")
    
    # Get topic-specific articles
    articles_per_topic = (target_count - len(all_articles)) // len(topics)
    
    for topic in topics:
        if len(all_articles) >= target_count:
            break
            
        print(f"Scraping articles for: {topic}")
        
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
            
            time.sleep(0.1)
        
        print(f"Scraped {topic_count} articles for {topic}")
    
    print(f"\nTotal articles scraped: {len(all_articles)}")
    
    documents = [article['content'] for article in all_articles]
    metadata = [{'title': article['title'], 'topic': article['topic']} for article in all_articles]
    
    return documents, metadata

def extract_cluster_keywords(documents, cluster_assignments, vectorizer, top_k=10):
    """Extract top keywords for each cluster with weights"""
    
    cluster_keywords = {}
    unique_clusters = sorted(list(set(cluster_assignments)))
    
    for cluster_id in unique_clusters:
        cluster_docs = [documents[i] for i, c in enumerate(cluster_assignments) if c == cluster_id]
        
        if not cluster_docs:
            continue
        
        all_other_docs = [documents[i] for i, c in enumerate(cluster_assignments) if c != cluster_id]
        
        if not all_other_docs:
            all_other_docs = cluster_docs
        
        cluster_text = ' '.join(cluster_docs)
        other_text = ' '.join(all_other_docs)
        
        temp_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = temp_vectorizer.fit_transform([cluster_text, other_text])
        
        feature_names = temp_vectorizer.get_feature_names_out()
        cluster_scores = tfidf_matrix[0].toarray()[0]
        
        top_indices = cluster_scores.argsort()[-top_k:][::-1]
        keywords = [(feature_names[i], cluster_scores[i]) for i in top_indices if cluster_scores[i] > 0]
        
        cluster_keywords[cluster_id] = keywords
    
    return cluster_keywords

def plot_pca_clusters(X, cluster_assignments, cluster_keywords, metadata):
    """Create PCA visualization with cluster overlays"""
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
    
    plt.figure(figsize=(16, 12))
    
    unique_clusters = sorted(list(set(cluster_assignments)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = np.array(cluster_assignments) == cluster_id
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.6, s=40)
    
    for cluster_id in unique_clusters:
        cluster_mask = np.array(cluster_assignments) == cluster_id
        if np.sum(cluster_mask) > 0:
            center_x = np.mean(X_pca[cluster_mask, 0])
            center_y = np.mean(X_pca[cluster_mask, 1])
            plt.scatter(center_x, center_y, c='black', marker='x', s=300, linewidth=4)
            
            if cluster_id in cluster_keywords and cluster_keywords[cluster_id]:
                top_words = [word for word, _ in cluster_keywords[cluster_id][:3]]
                plt.annotate(', '.join(top_words), (center_x, center_y), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    plt.title(f'CRP Wikipedia Document Clustering\n({len(unique_clusters)} clusters found)', fontsize=16, weight='bold')
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
            'sample_titles': cluster_titles[:5]
        }
    
    return cluster_topic_analysis

def save_crp_results_to_files(results, output_dir="crp_wikipedia_results"):
    """Save comprehensive CRP results to multiple file formats"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"\n=== SAVING CRP RESULTS TO {full_output_dir} ===")
    
    # 1. Save main results dataframe
    results_file = os.path.join(full_output_dir, "document_analysis.csv")
    results['results_dataframe'].to_csv(results_file, index=False)
    print(f"✓ Saved document analysis to: {results_file}")
    
    # 2. Save cluster assignments with document info
    cluster_file = os.path.join(full_output_dir, "cluster_assignments.csv")
    cluster_df = pd.DataFrame({
        'doc_id': range(len(results['documents'])),
        'title': [m['title'] for m in results['metadata']],
        'topic': [m['topic'] for m in results['metadata']],
        'cluster_id': results['cluster_assignments'],
        'document_preview': [doc[:200] + "..." if len(doc) > 200 else doc 
                           for doc in results['documents']]
    })
    cluster_df.to_csv(cluster_file, index=False)
    print(f"✓ Saved cluster assignments to: {cluster_file}")
    
    # 3. Save cluster keywords
    cluster_analysis_file = os.path.join(full_output_dir, "cluster_keywords.json")
    serializable_keywords = {}
    for cluster_id, keywords in results['cluster_keywords'].items():
        serializable_keywords[str(cluster_id)] = [
            {'word': word, 'weight': float(weight)} for word, weight in keywords
        ]
    
    with open(cluster_analysis_file, 'w') as f:
        json.dump(serializable_keywords, f, indent=2)
    print(f"✓ Saved cluster keywords to: {cluster_analysis_file}")
    
    # 4. Save detailed analysis
    detailed_analysis_file = os.path.join(full_output_dir, "detailed_cluster_analysis.txt")
    with open(detailed_analysis_file, 'w', encoding='utf-8') as f:
        f.write("=== CRP WIKIPEDIA CLUSTER ANALYSIS ===\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Documents: {len(results['documents'])}\n")
        f.write(f"Total Clusters: {results['crp_model'].n_clusters}\n")
        f.write(f"Alpha Parameter: {results['crp_model'].alpha}\n\n")
        
        f.write("=== CLUSTER DETAILS ===\n\n")
        for cluster_id in sorted(results['cluster_keywords'].keys()):
            cluster_size = results['cluster_assignments'].count(cluster_id)
            f.write(f"Cluster {cluster_id} ({cluster_size} documents):\n")
            
            keywords = [f"{word}({weight:.3f})" for word, weight in results['cluster_keywords'][cluster_id][:8]]
            f.write("  Top keywords: " + ", ".join(keywords) + "\n")
            
            cluster_docs = [i for i, c in enumerate(results['cluster_assignments']) if c == cluster_id]
            sample_titles = [results['metadata'][i]['title'] for i in cluster_docs[:5]]
            f.write(f"  Sample documents: {', '.join(sample_titles)}\n")
            
            if cluster_id in results['cluster_analysis']:
                topics = results['cluster_analysis'][cluster_id]['topic_distribution']
                f.write(f"  Topic distribution: {topics}\n")
            f.write("\n")
    
    print(f"✓ Saved detailed analysis to: {detailed_analysis_file}")
    
    # 5. Save raw documents
    documents_file = os.path.join(full_output_dir, "raw_documents.json")
    documents_data = []
    for i, (doc, meta) in enumerate(zip(results['documents'], results['metadata'])):
        documents_data.append({
            'doc_id': i,
            'title': meta['title'],
            'topic': meta['topic'],
            'cluster_id': results['cluster_assignments'][i],
            'content': doc
        })
    
    with open(documents_file, 'w', encoding='utf-8') as f:
        json.dump(documents_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved raw documents to: {documents_file}")
    
    # 6. Save model objects
    models_file = os.path.join(full_output_dir, "models.pkl")
    model_objects = {
        'crp_model': results['crp_model'],
        'vectorizer': results['vectorizer'],
        'pca_model': results['pca_model']
    }
    
    with open(models_file, 'wb') as f:
        pickle.dump(model_objects, f)
    print(f"✓ Saved model objects to: {models_file}")
    
    # 7. Save summary statistics
    summary_file = os.path.join(full_output_dir, "summary_statistics.json")
    cluster_sizes = Counter(results['cluster_assignments'])
    
    summary_stats = {
        'experiment_info': {
            'timestamp': timestamp,
            'total_documents': len(results['documents']),
            'total_clusters': int(results['crp_model'].n_clusters),
            'alpha_parameter': float(results['crp_model'].alpha)
        },
        'cluster_statistics': {
            'cluster_sizes': dict(cluster_sizes),
            'largest_cluster_size': int(max(cluster_sizes.values())),
            'smallest_cluster_size': int(min(cluster_sizes.values())),
            'avg_cluster_size': float(np.mean(list(cluster_sizes.values()))),
            'median_cluster_size': float(np.median(list(cluster_sizes.values())))
        },
        'topic_statistics': {
            'topic_counts': dict(Counter([m['topic'] for m in results['metadata']]))
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✓ Saved summary statistics to: {summary_file}")
    
    # 8. Create README
    readme_file = os.path.join(full_output_dir, "README.md")
    with open(readme_file, 'w') as f:
        f.write("# CRP Wikipedia Document Clustering Results\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Documents:** {len(results['documents'])}\n")
        f.write(f"**Total Clusters:** {results['crp_model'].n_clusters}\n")
        f.write(f"**Alpha Parameter:** {results['crp_model'].alpha}\n\n")
        
        f.write("## Files\n\n")
        f.write("- `document_analysis.csv`: Main results\n")
        f.write("- `cluster_assignments.csv`: Detailed assignments\n")
        f.write("- `cluster_keywords.json`: Keywords per cluster\n")
        f.write("- `detailed_cluster_analysis.txt`: Human-readable analysis\n")
        f.write("- `raw_documents.json`: Full document text\n")
        f.write("- `models.pkl`: Trained models\n")
        f.write("- `summary_statistics.json`: Key statistics\n\n")
        
        f.write("## Key Findings\n\n")
        largest_cluster = max(cluster_sizes, key=cluster_sizes.get)
        f.write(f"- Largest cluster: {largest_cluster} ({cluster_sizes[largest_cluster]} documents)\n")
        f.write(f"- Average cluster size: {np.mean(list(cluster_sizes.values())):.1f}\n")
        f.write(f"- Singleton clusters: {sum(1 for size in cluster_sizes.values() if size == 1)}\n")
    
    print(f"✓ Saved README to: {readme_file}")
    print(f"\n✓ All results saved to: {full_output_dir}")
    
    return full_output_dir

print("Before main")
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
    crp = ChineseRestaurantProcess(alpha=3.0)  # Lower alpha to reduce singletons
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
    
    results_df = results_df.sort_values('cluster_id')
    
    print("\n=== DOCUMENT ASSIGNMENTS ===")
    for cluster_id in sorted(results_df['cluster_id'].unique()):
        cluster_docs = results_df[results_df['cluster_id'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
        for _, row in cluster_docs.head(5).iterrows():
            print(f"  • {row['title']} [{row['topic']}]")
    
    # Extract cluster keywords
    print("\nExtracting cluster keywords...")
    cluster_keywords = extract_cluster_keywords(documents, crp.cluster_assignments, vectorizer, top_k=8)
    
    print("\n=== CLUSTER KEYWORDS ===")
    for cluster_id in sorted(cluster_keywords.keys()):
        print(f"\nCluster {cluster_id}:")
        for word, weight in cluster_keywords[cluster_id]:
            print(f"  {word}: {weight:.4f}")
    
    # Analyze cluster composition
    print("\n=== CLUSTER-TOPIC ANALYSIS ===")
    cluster_analysis = analyze_cluster_topics(crp.cluster_assignments, metadata)
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\nCluster {cluster_id} ({analysis['size']})")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    results = main()
