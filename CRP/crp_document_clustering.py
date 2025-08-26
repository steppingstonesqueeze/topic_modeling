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
        probs.append((new_cluster_id, crp_prob * 0.1))  # Base probability for new cluster
        
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

def generate_synthetic_documents(n_docs=500):
    """Generate 500 synthetic documents across different topics"""
    
    # Topic templates with key terms
    topics = {
        'technology': {
            'keywords': ['software', 'computer', 'algorithm', 'data', 'programming', 'AI', 'machine learning', 'cloud', 'digital', 'innovation'],
            'contexts': ['development', 'system', 'application', 'platform', 'solution', 'technology', 'implementation', 'framework']
        },
        'health': {
            'keywords': ['medical', 'health', 'patient', 'treatment', 'disease', 'medicine', 'doctor', 'hospital', 'therapy', 'diagnosis'],
            'contexts': ['care', 'wellness', 'recovery', 'prevention', 'symptoms', 'condition', 'procedure', 'healthcare']
        },
        'business': {
            'keywords': ['market', 'company', 'profit', 'customer', 'strategy', 'growth', 'revenue', 'investment', 'management', 'sales'],
            'contexts': ['business', 'industry', 'economy', 'finance', 'corporate', 'enterprise', 'organization', 'commercial']
        },
        'education': {
            'keywords': ['student', 'learning', 'school', 'teacher', 'education', 'knowledge', 'study', 'academic', 'curriculum', 'research'],
            'contexts': ['university', 'classroom', 'course', 'degree', 'skill', 'training', 'development', 'scholarship']
        },
        'environment': {
            'keywords': ['climate', 'environment', 'green', 'sustainable', 'energy', 'pollution', 'conservation', 'renewable', 'ecosystem', 'carbon'],
            'contexts': ['nature', 'planet', 'earth', 'global', 'environmental', 'ecology', 'sustainability', 'climate change']
        },
        'sports': {
            'keywords': ['team', 'player', 'game', 'sport', 'training', 'competition', 'athlete', 'performance', 'championship', 'fitness'],
            'contexts': ['league', 'tournament', 'match', 'season', 'victory', 'coaching', 'exercise', 'physical']
        }
    }
    
    documents = []
    doc_labels = []
    
    # Generate documents for each topic
    docs_per_topic = n_docs // len(topics)
    
    for topic_name, topic_data in topics.items():
        for _ in range(docs_per_topic):
            # Generate a short paragraph
            doc_length = random.randint(40, 95)  # 40-95 words
            words = []
            
            # Add topic-specific keywords
            n_keywords = random.randint(3, 7)
            topic_words = random.sample(topic_data['keywords'], n_keywords)
            context_words = random.sample(topic_data['contexts'], random.randint(2, 4))
            
            # Common filler words
            fillers = ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those', 'can', 'may', 'must', 'new', 'good', 'great', 'important', 'different', 'large', 'small', 'high', 'low', 'best', 'better', 'more', 'most', 'many', 'much', 'well', 'also', 'very', 'much', 'way', 'through', 'over', 'out', 'up', 'time', 'work', 'help', 'make', 'use', 'need', 'get', 'take', 'provide', 'include', 'show', 'find', 'give', 'come', 'go', 'know', 'think', 'see', 'look', 'want', 'become', 'seem', 'turn', 'part', 'problem', 'solution', 'process', 'result', 'example', 'information', 'number', 'year', 'day', 'people', 'person', 'group', 'world', 'country', 'state', 'place', 'right', 'public', 'social', 'national', 'local', 'community']
            
            all_words = topic_words + context_words
            
            # Fill remaining words
            while len(all_words) < doc_length:
                if random.random() < 0.3:  # 30% chance for topic words
                    all_words.append(random.choice(topic_words + context_words))
                else:
                    all_words.append(random.choice(fillers))
            
            # Create document
            random.shuffle(all_words)
            doc_text = ' '.join(all_words[:doc_length])
            documents.append(doc_text)
            doc_labels.append(topic_name)
    
    # Fill remaining documents with mixed topics
    remaining = n_docs - len(documents)
    for _ in range(remaining):
        # Mix multiple topics
        selected_topics = random.sample(list(topics.keys()), random.randint(2, 3))
        doc_length = random.randint(40, 95)
        words = []
        
        for topic_name in selected_topics:
            topic_data = topics[topic_name]
            words.extend(random.sample(topic_data['keywords'], random.randint(2, 4)))
            words.extend(random.sample(topic_data['contexts'], random.randint(1, 2)))
        
        # Fill with common words
        fillers = ['the', 'and', 'of', 'to', 'in', 'for', 'with', 'on', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that', 'can', 'may', 'new', 'good', 'important', 'different', 'way', 'work', 'help', 'make', 'use', 'need', 'get', 'provide', 'show', 'people', 'time', 'process', 'system']
        
        while len(words) < doc_length:
            words.append(random.choice(fillers))
        
        random.shuffle(words)
        doc_text = ' '.join(words[:doc_length])
        documents.append(doc_text)
        doc_labels.append('mixed')
    
    return documents[:n_docs], doc_labels[:n_docs]

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

def plot_pca_clusters(X, cluster_assignments, cluster_keywords, title="CRP Document Clustering"):
    """Create PCA visualization with cluster overlays"""
    
    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Color palette
    unique_clusters = sorted(list(set(cluster_assignments)))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = np.array(cluster_assignments) == cluster_id
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=50)
    
    # Add cluster centers
    for cluster_id in unique_clusters:
        cluster_mask = np.array(cluster_assignments) == cluster_id
        if np.sum(cluster_mask) > 0:
            center_x = np.mean(X_pca[cluster_mask, 0])
            center_y = np.mean(X_pca[cluster_mask, 1])
            plt.scatter(center_x, center_y, c='black', marker='x', s=200, linewidth=3)
            
            # Add top keywords as text
            if cluster_id in cluster_keywords and cluster_keywords[cluster_id]:
                top_words = [word for word, _ in cluster_keywords[cluster_id][:3]]
                plt.annotate(', '.join(top_words), (center_x, center_y), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.title(f'{title}\n({len(unique_clusters)} clusters found)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return X_pca, pca

def main():
    """Main execution function"""
    print("Generating 500 synthetic documents...")
    documents, true_labels = generate_synthetic_documents(500)
    
    print(f"Generated {len(documents)} documents")
    print(f"Sample document: {documents[0][:100]}...")
    
    # Vectorize documents
    print("\nVectorizing documents with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english', max_df=0.7, min_df=3, ngram_range=(1, 2))
    X = vectorizer.fit_transform(documents)
    print(f"Document matrix shape: {X.shape}")
    
    # Run Chinese Restaurant Process
    print("\nRunning Chinese Restaurant Process...")
    crp = ChineseRestaurantProcess(alpha=5.0)  # Much higher alpha = more clusters
    crp.fit(X, n_iterations=120)
    
    print(f"\nCRP found {crp.n_clusters} clusters")
    print(f"Cluster sizes: {Counter(crp.cluster_assignments)}")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'doc_id': range(len(documents)),
        'cluster_id': crp.cluster_assignments,
        'document_text': documents,
        'true_label': true_labels
    })
    
    # Sort by cluster ID
    results_df = results_df.sort_values('cluster_id')
    
    print("\n=== DOCUMENT ASSIGNMENTS (sorted by cluster ID) ===")
    for cluster_id in sorted(results_df['cluster_id'].unique()):
        cluster_docs = results_df[results_df['cluster_id'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
        for _, row in cluster_docs.head(3).iterrows():  # Show first 3 docs per cluster
            print(f"  Doc {row['doc_id']}: {row['document_text'][:80]}...")
    
    # Extract cluster keywords
    print("\nExtracting cluster keywords...")
    cluster_keywords = extract_cluster_keywords(documents, crp.cluster_assignments, vectorizer)
    
    print("\n=== CLUSTER KEYWORDS WITH WEIGHTS ===")
    for cluster_id in sorted(cluster_keywords.keys()):
        print(f"\nCluster {cluster_id}:")
        for word, weight in cluster_keywords[cluster_id]:
            print(f"  {word}: {weight:.4f}")
    
    # Create PCA visualization
    print("\nCreating PCA visualization...")
    X_pca, pca_model = plot_pca_clusters(X, crp.cluster_assignments, cluster_keywords)
    
    print("\n=== SUMMARY ===")
    print(f"Total documents: {len(documents)}")
    print(f"Number of clusters found: {crp.n_clusters}")
    print(f"Alpha parameter used: {crp.alpha}")
    print(f"PCA explained variance: PC1={pca_model.explained_variance_ratio_[0]:.2%}, PC2={pca_model.explained_variance_ratio_[1]:.2%}")
    
    return {
        'documents': documents,
        'cluster_assignments': crp.cluster_assignments,
        'cluster_keywords': cluster_keywords,
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