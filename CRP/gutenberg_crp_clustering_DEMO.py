import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap
from collections import defaultdict, Counter
import random
import re
import requests
import json
import pickle
import os
from datetime import datetime
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
        
    def fit(self, X, n_iterations=200, verbose=True):
        """
        Fit CRP model to document vectors with progress reporting
        
        Args:
            X: Document-term matrix (n_docs x n_features)
            n_iterations: Number of Gibbs sampling iterations
            verbose: Print progress
        """
        n_docs, n_features = X.shape
        
        if verbose:
            print(f"Initializing CRP for {n_docs} documents with {n_features} features")
        
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
        
        # Gibbs sampling with progress reporting
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
            
            if verbose and (iteration % 40 == 0 or iteration == n_iterations - 1):
                current_clusters = len(set(self.cluster_assignments))
                cluster_sizes = Counter(self.cluster_assignments)
                largest_cluster = max(cluster_sizes.values())
                singletons = sum(1 for size in cluster_sizes.values() if size == 1)
                
                print(f"Iteration {iteration:3d}: {current_clusters:2d} clusters "
                      f"(largest: {largest_cluster}, singletons: {singletons})")
        
        # Clean up empty clusters and renumber
        self._cleanup_clusters()
        
        if verbose:
            print(f"Final: {self.n_clusters} clusters discovered")
    
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
            
            # Likelihood (cosine similarity to cluster center)
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
        probs.append((new_cluster_id, crp_prob * 0.3))  # Base probability for new cluster
        
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

class GutenbergCorpusBuilder:
    def __init__(self):
        self.base_url = "https://www.gutenberg.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CRP-Gutenberg-Analysis/1.0 (Educational Research)'
        })
        
    def get_top_authors(self, n_authors=50):
        """Get top N most popular authors from Project Gutenberg"""
        # This is a curated list of top authors - in practice you'd scrape the popular page
        top_authors = [
            "Shakespeare, William", "Dickens, Charles", "Doyle, Arthur Conan",
            "Twain, Mark", "Austen, Jane", "Carroll, Lewis", "Wilde, Oscar", 
            "Verne, Jules", "Wells, H. G.", "Shelley, Mary Wollstonecraft",
            "Stoker, Bram", "Poe, Edgar Allan", "London, Jack", "Stevenson, Robert Louis",
            "Kipling, Rudyard", "Conrad, Joseph", "Hardy, Thomas", "Gaskell, Elizabeth Cleghorn",
            "Trollope, Anthony", "Scott, Walter", "Eliot, George", "Thackeray, William Makepeace",
            "Dumas, Alexandre", "Balzac, Honoré de", "Hugo, Victor", "Zola, Émile",
            "Dostoyevsky, Fyodor", "Tolstoy, Leo", "Turgenev, Ivan Sergeevich", 
            "Chekhov, Anton Pavlovich", "Ibsen, Henrik", "Strindberg, August",
            "Goethe, Johann Wolfgang von", "Schiller, Friedrich", "Nietzsche, Friedrich Wilhelm",
            "Darwin, Charles", "Mill, John Stuart", "James, Henry", "Wharton, Edith",
            "Alcott, Louisa May", "Hawthorne, Nathaniel", "Melville, Herman", 
            "Whitman, Walt", "Thoreau, Henry David", "Emerson, Ralph Waldo",
            "Irving, Washington", "Cooper, James Fenimore", "Bierce, Ambrose", "Crane, Stephen"
        ]
        
        return top_authors[:n_authors]
    
    def search_author_works(self, author_name, max_books=5):
        """
        Search for works by a specific author
        Returns a list of book metadata
        """
        # This is a simplified implementation - in practice you'd use the Gutenberg API
        # For demonstration, we'll use a mapping of known popular works
        author_works = {
            "Shakespeare, William": [
                {"title": "Hamlet", "id": "1524", "url": "https://www.gutenberg.org/files/1524/1524-0.txt"},
                {"title": "Romeo and Juliet", "id": "1513", "url": "https://www.gutenberg.org/files/1513/1513-0.txt"},
                {"title": "Macbeth", "id": "1533", "url": "https://www.gutenberg.org/files/1533/1533-0.txt"},
                {"title": "Othello", "id": "1531", "url": "https://www.gutenberg.org/files/1531/1531-0.txt"},
                {"title": "King Lear", "id": "1532", "url": "https://www.gutenberg.org/files/1532/1532-0.txt"}
            ],
            "Dickens, Charles": [
                {"title": "A Christmas Carol", "id": "19337", "url": "https://www.gutenberg.org/files/19337/19337-0.txt"},
                {"title": "Oliver Twist", "id": "730", "url": "https://www.gutenberg.org/files/730/730-0.txt"},
                {"title": "Great Expectations", "id": "1400", "url": "https://www.gutenberg.org/files/1400/1400-0.txt"},
                {"title": "David Copperfield", "id": "766", "url": "https://www.gutenberg.org/files/766/766-0.txt"},
                {"title": "A Tale of Two Cities", "id": "98", "url": "https://www.gutenberg.org/files/98/98-0.txt"}
            ],
            "Austen, Jane": [
                {"title": "Pride and Prejudice", "id": "1342", "url": "https://www.gutenberg.org/files/1342/1342-0.txt"},
                {"title": "Emma", "id": "158", "url": "https://www.gutenberg.org/files/158/158-0.txt"},
                {"title": "Sense and Sensibility", "id": "161", "url": "https://www.gutenberg.org/files/161/161-0.txt"},
                {"title": "Mansfield Park", "id": "141", "url": "https://www.gutenberg.org/files/141/141-0.txt"},
                {"title": "Persuasion", "id": "105", "url": "https://www.gutenberg.org/files/105/105-0.txt"}
            ]
            # Add more authors as needed - this is a demonstration subset
        }
        
        return author_works.get(author_name, [])[:max_books]
    
    def download_book_text(self, book_info):
        """Download and clean text from a Gutenberg book"""
        try:
            response = self.session.get(book_info["url"], timeout=30)
            response.raise_for_status()
            text = response.text
            
            # Clean Gutenberg text
            text = self.clean_gutenberg_text(text)
            
            # Return meaningful chunk (avoid massive texts)
            if len(text) > 50000:  # Limit to ~50k characters
                text = text[:50000]
            
            return text
        except Exception as e:
            print(f"Error downloading {book_info['title']}: {e}")
            return None
    
    def clean_gutenberg_text(self, text):
        """Clean Project Gutenberg text"""
        if not text:
            return ""
        
        # Remove Gutenberg header/footer
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find start of actual content
        for i, line in enumerate(lines):
            if 'START OF THE PROJECT GUTENBERG' in line.upper() or \
               'START OF THIS PROJECT GUTENBERG' in line.upper():
                start_idx = i + 1
                break
        
        # Find end of actual content
        for i, line in enumerate(lines[start_idx:], start_idx):
            if 'END OF THE PROJECT GUTENBERG' in line.upper() or \
               'END OF THIS PROJECT GUTENBERG' in line.upper():
                end_idx = i
                break
        
        text = '\n'.join(lines[start_idx:end_idx])
        
        # Basic cleaning
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def build_corpus(self, authors, books_per_author=3, min_text_length=5000):
        """Build a corpus from multiple authors"""
        print(f"Building corpus from {len(authors)} authors ({books_per_author} books each)")
        
        all_documents = []
        all_metadata = []
        
        for i, author in enumerate(authors, 1):
            print(f"\n[{i:2d}/{len(authors)}] Processing {author}...")
            
            # Get works for this author
            works = self.search_author_works(author, max_books=books_per_author)
            
            if not works:
                print(f"  No works found for {author}")
                continue
            
            author_docs = 0
            for work in works:
                print(f"  Downloading: {work['title']}")
                text = self.download_book_text(work)
                
                if text and len(text) >= min_text_length:
                    all_documents.append(text)
                    all_metadata.append({
                        'author': author,
                        'title': work['title'],
                        'gutenberg_id': work['id'],
                        'text_length': len(text)
                    })
                    author_docs += 1
                    print(f"    Added ({len(text):,} chars)")
                else:
                    print(f"    Skipped (too short or failed)")
                
                time.sleep(0.5)  # Rate limiting
            
            print(f"  Added {author_docs} books for {author}")
        
        print(f"\nCorpus complete: {len(all_documents)} books from {len(set(m['author'] for m in all_metadata))} authors")
        
        return all_documents, all_metadata

def extract_cluster_keywords(documents, cluster_assignments, vectorizer, top_k=10):
    """Extract distinctive keywords for each cluster"""
    cluster_keywords = {}
    unique_clusters = sorted(list(set(cluster_assignments)))
    
    for cluster_id in unique_clusters:
        cluster_docs = [documents[i] for i, c in enumerate(cluster_assignments) if c == cluster_id]
        
        if not cluster_docs:
            continue
        
        # All other documents
        other_docs = [documents[i] for i, c in enumerate(cluster_assignments) if c != cluster_id]
        
        if not other_docs:
            other_docs = cluster_docs
        
        # Create cluster vs others comparison
        cluster_text = ' '.join(cluster_docs)
        other_text = ' '.join(other_docs)
        
        # Use more permissive parameters for small document sets
        temp_vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words='english', 
            max_df=0.95,  # More permissive
            min_df=1,     # Allow single occurrences
            ngram_range=(1, 1)  # Only unigrams for stability
        )
        
        try:
            tfidf_matrix = temp_vectorizer.fit_transform([cluster_text, other_text])
            
            feature_names = temp_vectorizer.get_feature_names_out()
            cluster_scores = tfidf_matrix[0].toarray()[0]
            
            # Get top distinctive terms
            top_indices = cluster_scores.argsort()[-top_k:][::-1]
            keywords = [(feature_names[i], cluster_scores[i]) for i in top_indices if cluster_scores[i] > 0]
            
            cluster_keywords[cluster_id] = keywords
            
        except ValueError as e:
            # Fallback: use simple word frequency for problematic clusters
            print(f"  Warning: Using word frequency fallback for cluster {cluster_id}")
            
            words = cluster_text.lower().split()
            words = [w for w in words if w.isalpha() and len(w) > 2]
            word_counts = Counter(words)
            
            keywords = [(word, count) for word, count in word_counts.most_common(top_k)]
            cluster_keywords[cluster_id] = keywords
    
    return cluster_keywords

def create_visualizations(X, cluster_assignments, metadata, cluster_keywords, output_dir):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    # 1. PCA visualization
    print("  Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
    
    plt.figure(figsize=(16, 12))
    unique_clusters = sorted(list(set(cluster_assignments)))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = np.array(cluster_assignments) == cluster_id
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                   c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=50)
    
    # Add cluster centers and labels
    for cluster_id in unique_clusters:
        cluster_mask = np.array(cluster_assignments) == cluster_id
        if np.sum(cluster_mask) > 0:
            center_x = np.mean(X_pca[cluster_mask, 0])
            center_y = np.mean(X_pca[cluster_mask, 1])
            plt.scatter(center_x, center_y, c='black', marker='x', s=400, linewidth=4)
            
            # Add keywords
            if cluster_id in cluster_keywords and cluster_keywords[cluster_id]:
                top_words = [word for word, _ in cluster_keywords[cluster_id][:3]]
                plt.annotate(', '.join(top_words), (center_x, center_y), 
                           xytext=(8, 8), textcoords='offset points', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    plt.title('CRP Clustering of Project Gutenberg Books\n(PCA Projection)', fontsize=16, weight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    
    # Legend handling for many clusters
    if len(unique_clusters) > 20:
        plt.legend().set_visible(False)
        plt.text(0.02, 0.98, f'{len(unique_clusters)} clusters found', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_clusters.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. UMAP visualization (better for large datasets)
    print("  Computing UMAP...")
    try:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        
        plt.figure(figsize=(16, 12))
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = np.array(cluster_assignments) == cluster_id
            plt.scatter(X_umap[cluster_mask, 0], X_umap[cluster_mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', alpha=0.7, s=50)
        
        plt.title('CRP Clustering of Project Gutenberg Books\n(UMAP Projection)', 
                 fontsize=16, weight='bold')
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        
        if len(unique_clusters) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_clusters.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("  UMAP not available, skipping UMAP visualization")
    
    # 3. Cluster composition by author
    print("  Creating author composition plot...")
    cluster_author_data = []
    for i, meta in enumerate(metadata):
        cluster_author_data.append({
            'cluster': cluster_assignments[i],
            'author': meta['author'].split(',')[0],  # Last name only
            'title': meta['title']
        })
    
    cluster_author_df = pd.DataFrame(cluster_author_data)
    
    # Create author-cluster heatmap
    author_cluster_matrix = cluster_author_df.groupby(['author', 'cluster']).size().unstack(fill_value=0)
    
    if author_cluster_matrix.shape[0] <= 30 and author_cluster_matrix.shape[1] <= 20:
        plt.figure(figsize=(14, 10))
        sns.heatmap(author_cluster_matrix, annot=True, fmt='d', cmap='YlOrRd', 
                   cbar_kws={'label': 'Number of Books'})
        plt.title('Author Distribution Across Clusters', fontsize=14, weight='bold')
        plt.xlabel('Cluster ID')
        plt.ylabel('Author')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'author_cluster_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    return X_pca, pca

def analyze_clusters(cluster_assignments, metadata, cluster_keywords):
    """Analyze cluster composition and characteristics"""
    cluster_analysis = {}
    unique_clusters = sorted(list(set(cluster_assignments)))
    
    for cluster_id in unique_clusters:
        cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
        cluster_metadata = [metadata[i] for i in cluster_indices]
        
        # Author distribution
        authors = [meta['author'] for meta in cluster_metadata]
        author_counts = Counter(authors)
        
        # Average text length
        text_lengths = [meta['text_length'] for meta in cluster_metadata]
        avg_length = np.mean(text_lengths)
        
        # Sample titles
        sample_titles = [f"{meta['title']} ({meta['author'].split(',')[0]})" 
                        for meta in cluster_metadata[:5]]
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_indices),
            'authors': dict(author_counts.most_common()),
            'avg_text_length': avg_length,
            'sample_works': sample_titles,
            'top_keywords': [word for word, _ in cluster_keywords.get(cluster_id, [])[:8]]
        }
    
    return cluster_analysis

def save_results(documents, metadata, cluster_assignments, cluster_keywords, 
                cluster_analysis, crp_model, vectorizer, output_dir):
    """Save comprehensive results to files"""
    print("Saving results to files...")
    
    # 1. Main results CSV
    results_df = pd.DataFrame({
        'doc_id': range(len(documents)),
        'cluster_id': cluster_assignments,
        'author': [m['author'] for m in metadata],
        'title': [m['title'] for m in metadata],
        'gutenberg_id': [m['gutenberg_id'] for m in metadata],
        'text_length': [m['text_length'] for m in metadata]
    })
    
    results_df.to_csv(os.path.join(output_dir, 'clustering_results.csv'), index=False)
    
    # 2. Cluster keywords
    with open(os.path.join(output_dir, 'cluster_keywords.json'), 'w') as f:
        serializable_keywords = {}
        for cluster_id, keywords in cluster_keywords.items():
            serializable_keywords[str(cluster_id)] = [
                {'word': word, 'weight': float(weight)} for word, weight in keywords
            ]
        json.dump(serializable_keywords, f, indent=2)
    
    # 3. Cluster analysis
    with open(os.path.join(output_dir, 'cluster_analysis.json'), 'w') as f:
        serializable_analysis = {}
        for cluster_id, analysis in cluster_analysis.items():
            serializable_analysis[str(cluster_id)] = {
                'size': analysis['size'],
                'authors': analysis['authors'],
                'avg_text_length': float(analysis['avg_text_length']),
                'sample_works': analysis['sample_works'],
                'top_keywords': analysis['top_keywords']
            }
        json.dump(serializable_analysis, f, indent=2)
    
    # 4. Model objects
    model_objects = {
        'crp_model': crp_model,
        'vectorizer': vectorizer,
        'metadata': metadata
    }
    
    with open(os.path.join(output_dir, 'models.pkl'), 'wb') as f:
        pickle.dump(model_objects, f)
    
    # 5. Summary report
    with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("CRP CLUSTERING OF PROJECT GUTENBERG BOOKS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Books: {len(documents)}\n")
        f.write(f"Unique Authors: {len(set(m['author'] for m in metadata))}\n")
        f.write(f"Clusters Found: {crp_model.n_clusters}\n")
        f.write(f"Alpha Parameter: {crp_model.alpha}\n\n")
        
        # Cluster details
        f.write("CLUSTER DETAILS\n")
        f.write("-" * 20 + "\n\n")
        
        for cluster_id in sorted(cluster_analysis.keys()):
            analysis = cluster_analysis[cluster_id]
            f.write(f"Cluster {cluster_id} ({analysis['size']} books):\n")
            f.write(f"  Keywords: {', '.join(analysis['top_keywords'])}\n")
            f.write(f"  Top Authors: {', '.join([f'{auth} ({count})' for auth, count in list(analysis['authors'].items())[:3]])}\n")
            f.write(f"  Sample Works: {'; '.join(analysis['sample_works'][:3])}\n")
            f.write(f"  Avg Length: {analysis['avg_text_length']:,.0f} chars\n\n")
    
    print(f"All results saved to: {output_dir}")

def main():
    """Main execution function"""
    print("=== Large-Scale CRP Clustering for Project Gutenberg ===")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"gutenberg_crp_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize corpus builder
    corpus_builder = GutenbergCorpusBuilder()
    
    # Get top authors (start with subset for demonstration)
    top_authors = corpus_builder.get_top_authors(n_authors=10)  # Start with 10 for testing
    print(f"\nTargeted authors: {', '.join(top_authors)}")
    
    # Build corpus
    documents, metadata = corpus_builder.build_corpus(
        top_authors, 
        books_per_author=3,
        min_text_length=8000
    )
    
    if len(documents) < 5:
        print("Not enough documents collected. Exiting.")
        return
    
    print(f"\nCorpus statistics:")
    print(f"  Total books: {len(documents)}")
    print(f"  Unique authors: {len(set(m['author'] for m in metadata))}")
    print(f"  Average text length: {np.mean([m['text_length'] for m in metadata]):,.0f} chars")
    
    # Vectorize documents
    print("\nVectorizing documents with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        max_df=0.8,
        min_df=2,
        ngram_range=(1, 2),
        lowercase=True
    )
    
    X = vectorizer.fit_transform(documents)
    print(f"Document-term matrix: {X.shape[0]} documents x {X.shape[1]} features")
    
    # Run CRP clustering
    print(f"\nRunning Chinese Restaurant Process...")
    crp = ChineseRestaurantProcess(alpha=2.5)  # Moderate alpha for literary clustering
    crp.fit(X, n_iterations=250, verbose=True)
    
    # Analyze results
    print(f"\nCRP Results:")
    cluster_sizes = Counter(crp.cluster_assignments)
    print(f"  Clusters found: {crp.n_clusters}")
    print(f"  Largest cluster: {max(cluster_sizes.values())} books")
    print(f"  Singleton clusters: {sum(1 for size in cluster_sizes.values() if size == 1)}")
    print(f"  Average cluster size: {np.mean(list(cluster_sizes.values())):.1f}")
    
    # Extract cluster characteristics
    print("\nExtracting cluster keywords...")
    cluster_keywords = extract_cluster_keywords(documents, crp.cluster_assignments, vectorizer)
    
    print("\nAnalyzing cluster composition...")
    cluster_analysis = analyze_clusters(crp.cluster_assignments, metadata, cluster_keywords)
    
    # Print cluster summaries
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS")
    print("="*60)
    
    for cluster_id in sorted(cluster_analysis.keys()):
        analysis = cluster_analysis[cluster_id]
        print(f"\nCluster {cluster_id} ({analysis['size']} books):")
        print(f"  Keywords: {', '.join(analysis['top_keywords'][:5])}")
        
        # Show top authors
        top_authors_in_cluster = list(analysis['authors'].items())[:3]
        author_str = ', '.join([f"{auth.split(',')[0]} ({count})" for auth, count in top_authors_in_cluster])
        print(f"  Authors: {author_str}")
        
        # Show sample works
        print(f"  Sample works: {'; '.join(analysis['sample_works'][:3])}")
    
    # Create visualizations
    print(f"\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    X_pca, pca_model = create_visualizations(X, crp.cluster_assignments, metadata, 
                                           cluster_keywords, output_dir)
    
    # Save all results
    print(f"\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    save_results(documents, metadata, crp.cluster_assignments, cluster_keywords,
                cluster_analysis, crp, vectorizer, output_dir)
    
    # Final summary
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Books analyzed: {len(documents)}")
    print(f"Authors represented: {len(set(m['author'] for m in metadata))}")
    print(f"Clusters discovered: {crp.n_clusters}")
    print(f"Results saved to: {output_dir}")
    
    # Show most interesting clusters
    print(f"\nMost diverse clusters (by author count):")
    cluster_diversity = [(cid, len(analysis['authors'])) for cid, analysis in cluster_analysis.items()]
    cluster_diversity.sort(key=lambda x: x[1], reverse=True)
    
    for cluster_id, author_count in cluster_diversity[:5]:
        analysis = cluster_analysis[cluster_id]
        print(f"  Cluster {cluster_id}: {author_count} authors, {analysis['size']} books")
        print(f"    Keywords: {', '.join(analysis['top_keywords'][:4])}")
    
    return {
        'documents': documents,
        'metadata': metadata,
        'cluster_assignments': crp.cluster_assignments,
        'cluster_keywords': cluster_keywords,
        'cluster_analysis': cluster_analysis,
        'crp_model': crp,
        'vectorizer': vectorizer,
        'pca_coordinates': X_pca,
        'pca_model': pca_model,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Starting large-scale Gutenberg CRP analysis...")
    results = main()
