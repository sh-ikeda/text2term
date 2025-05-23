import os
import sys
import text2term
import owlready2
import pandas as pd
import ssl
from urllib import request
from text2term.term import OntologyTermType
from text2term.mapper import Mapper
from shutil import rmtree

# Default cache folder - can be modified but maintains backward compatibility
DEFAULT_CACHE_FOLDER = "cache"

"""
CACHING FUNCTIONS -- Public
"""

# Configure SSL context globally to ignore certificate validation
def disable_ssl_verification():
    """
    Disable SSL certificate verification globally for the application.
    This should be called early in your application's initialization.
    
    Warning: This makes HTTPS connections less secure, but may be necessary
    on some systems with outdated certificate stores.
    """
    # Create an unverified SSL context
    ssl_context = ssl._create_unverified_context()
    
    # Save the original urlopen function
    original_urlopen = request.urlopen
    
    # Create a new urlopen function that adds the ssl_context
    def patched_urlopen(url, *args, **kwargs):
        # Add the SSL context if not already provided
        if 'context' not in kwargs:
            kwargs['context'] = ssl_context
        return original_urlopen(url, *args, **kwargs)
    
    # Replace the urlopen function with our patched version
    request.urlopen = patched_urlopen

# Call this immediately to fix SSL issues
disable_ssl_verification()

# Set the default cache folder
def set_cache_folder(folder_path):
    """
    Set the default cache folder path.
    
    Parameters
    ----------
    folder_path : str
        Path to the new default cache folder
    
    Returns
    -------
    str
        The path to the new default cache folder
    """
    global DEFAULT_CACHE_FOLDER
    DEFAULT_CACHE_FOLDER = folder_path
    # Create the cache directory if it doesn't exist
    os.makedirs(DEFAULT_CACHE_FOLDER, exist_ok=True)
    return DEFAULT_CACHE_FOLDER

# Get the current default cache folder
def get_cache_folder():
    """
    Get the current default cache folder path.
    
    Returns
    -------
    str
        The path to the current default cache folder
    """
    return DEFAULT_CACHE_FOLDER

# Caches many ontologies from a csv
def cache_ontology_set(ontology_registry_path, cache_folder=None):
    """
    Cache ontologies from a CSV file.
    
    Parameters
    ----------
    ontology_registry_path : str
        Path to the CSV file containing ontology information
    cache_folder : str, optional
        Path to the folder where ontologies will be cached
    
    Returns
    -------
    dict
        A dictionary of ontology caches
    """
    # Use the provided cache_folder or fall back to the default
    cache_folder = cache_folder if cache_folder is not None else DEFAULT_CACHE_FOLDER
    
    # Create the cache directory if it doesn't exist
    os.makedirs(cache_folder, exist_ok=True)
    
    # Ensure the registry path exists before attempting to read it
    if not os.path.exists(ontology_registry_path):
        raise FileNotFoundError(f"Ontology registry file not found: {ontology_registry_path}")
    
    registry = pd.read_csv(ontology_registry_path)
    cache_set = {}
    for index, row in registry.iterrows():
        try:
            cache = text2term.cache_ontology(row.url, row.acronym, cache_folder=cache_folder)
            cache_set.update({row.acronym: cache})
        except Exception as err:
            err_message = f"Could not cache ontology {row.acronym} due to error: {str(err)}\n"
            sys.stderr.write(err_message)
        owlready2.default_world.ontologies.clear()
    return cache_set

# Will check if an acronym exists in the cache
def cache_exists(ontology_acronym='', cache_folder=None):
    """
    Check if an ontology acronym exists in the cache.
    
    Parameters
    ----------
    ontology_acronym : str
        The acronym of the ontology to check
    cache_folder : str, optional
        Path to the folder where ontologies are cached
    
    Returns
    -------
    bool
        True if the ontology cache exists, False otherwise
    """
    # Use the provided cache_folder or fall back to the default
    cache_folder = cache_folder if cache_folder is not None else DEFAULT_CACHE_FOLDER
    
    return os.path.exists(os.path.join(cache_folder, ontology_acronym))

# Clears the cache
def clear_cache(ontology_acronym='', cache_folder=None):
    """
    Clear the cache for a specific ontology or the entire cache.
    
    Parameters
    ----------
    ontology_acronym : str
        The acronym of the ontology to clear, or empty to clear all
    cache_folder : str, optional
        Path to the folder where ontologies are cached
    """
    # Use the provided cache_folder or fall back to the default
    cache_folder = cache_folder if cache_folder is not None else DEFAULT_CACHE_FOLDER
    
    cache_dir = cache_folder
    if ontology_acronym != '':
        cache_dir = os.path.join(cache_folder, ontology_acronym)
    # Is equivalent to: rm -r cache_dir
    try:
        if os.path.exists(cache_dir):
            rmtree(cache_dir)
            sys.stderr.write("Cache has been cleared successfully\n")
        else:
            sys.stderr.write(f"Cache directory {cache_dir} does not exist\n")
    except OSError as error:
        sys.stderr.write("Cache cannot be removed:")
        sys.stderr.write(str(error))

# Class that is returned to run
class OntologyCache:
    def __init__(self, ontology_acronym, cache_folder=None):
        """
        Initialize an OntologyCache object.
        
        Parameters
        ----------
        ontology_acronym : str
            The acronym of the ontology
        cache_folder : str, optional
            Path to the folder where ontologies are cached
        """
        self.acronym = ontology_acronym
        # Use the provided cache_folder or fall back to the default
        self.cache_folder = cache_folder if cache_folder is not None else DEFAULT_CACHE_FOLDER
        self.ontology = os.path.join(self.cache_folder, ontology_acronym)
        
        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_folder, exist_ok=True)
    
    def map_terms(self, source_terms, base_iris=(), excl_deprecated=False, max_mappings=3, min_score=0.3,
                  mapper=Mapper.TFIDF, output_file='', save_graphs=False, save_mappings=False, source_terms_ids=(),
                  term_type=OntologyTermType.CLASS):
        """
        Map terms to the ontology.
        
        Parameters
        ----------
        (see text2term documentation for parameter details)
        
        Returns
        -------
        DataFrame
            A pandas DataFrame containing the mapping results
        """
        return text2term.map_terms(source_terms, self.acronym, base_iris=base_iris,
                                   excl_deprecated=excl_deprecated, max_mappings=max_mappings, min_score=min_score,
                                   mapper=mapper, output_file=output_file, save_graphs=save_graphs,
                                   save_mappings=save_mappings, source_terms_ids=source_terms_ids, use_cache=True,
                                   term_type=term_type, cache_folder=self.cache_folder)
    
    def clear_cache(self):
        """Clear the cache for this ontology."""
        clear_cache(self.acronym, self.cache_folder)
    
    def cache_exists(self):
        """Check if the cache exists for this ontology."""
        return cache_exists(self.acronym, self.cache_folder)
    
    def acronym(self):
        """Get the acronym for this ontology."""
        return self.acronym
    