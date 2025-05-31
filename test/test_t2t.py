import os
import uuid
import tempfile
import unittest
import pandas as pd
import text2term
import text2term.onto_cache as onto_cache
from text2term import OntologyTermType, onto_utils
from text2term import Mapper
from text2term import OntologyTermCollector
from text2term.bioportal_mapper import BioPortalAnnotatorMapper
from text2term.zooma_mapper import ZoomaMapper

pd.set_option('display.max_columns', None)


class Text2TermTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(Text2TermTestSuite, cls).setUpClass()
        print("Setting up test suite global variables...")
        cls.EFO_URL = "https://github.com/EBISPOT/efo/releases/download/v3.57.0/efo.owl"
        cls.SOURCE_TERM_ID_COLUMN = "Source Term ID"
        cls.MAPPED_TERM_CURIE_COLUMN = "Mapped Term CURIE"
        cls.MAPPING_SCORE_COLUMN = "Mapping Score"
        cls.TAGS_COLUMN = "Tags"
        cls.TEST_CACHE_FOLDER = ".test_cache"
        
        # Configure the SSL verification to be disabled
        if hasattr(onto_cache, 'disable_ssl_verification'):
            onto_cache.disable_ssl_verification()
        
        # Ensure test environment is set up
        cls.setup_test_environment()

    @classmethod
    def tearDownClass(cls):
        super(Text2TermTestSuite, cls).tearDownClass()
        text2term.clear_cache(cache_folder=cls.TEST_CACHE_FOLDER)
        test_output_file_without_metadata = "test_output_without_metadata.csv"
        if os.path.exists(test_output_file_without_metadata):
            os.remove(test_output_file_without_metadata)
        test_output_file_with_metadata = "test_output_with_metadata.csv"
        if os.path.exists(test_output_file_with_metadata):
            os.remove("test_output_with_metadata.csv")

    @classmethod
    def setup_test_environment(cls):
        """Set up necessary test files and directories"""
        print("Setting up test environment...")
        
        # Create test cache folder
        if not os.path.exists(cls.TEST_CACHE_FOLDER):
            os.makedirs(cls.TEST_CACHE_FOLDER)
            print(f"Created test cache folder: {cls.TEST_CACHE_FOLDER}")
        
        # Create resources directory - try different possible paths
        resources_paths = [
            os.path.join("text2term", "resources"),
            os.path.join("..", "text2term", "resources")
        ]
        
        resources_dir = None
        for path in resources_paths:
            if os.path.exists(path):
                resources_dir = path
                break
        
        # If no existing resources directory found, create one
        if resources_dir is None:
            resources_dir = resources_paths[0]  # Default to first option
            os.makedirs(resources_dir, exist_ok=True)
            print(f"Created resources directory: {resources_dir}")
        
        # Create a minimal ontologies.csv file if it doesn't exist
        ontologies_file = os.path.join(resources_dir, "ontologies.csv")
        if not os.path.exists(ontologies_file):
            with open(ontologies_file, 'w') as f:
                f.write("acronym,url\n")
                f.write("EFO,https://github.com/EBISPOT/efo/releases/download/v3.57.0/efo.owl\n")
                f.write("CLO,http://purl.obolibrary.org/obo/clo.owl\n")
            print(f"Created sample ontologies file: {ontologies_file}")
        
        # Create simple_preprocess.txt for the preprocessing test
        preprocess_file = "simple_preprocess.txt"
        if not os.path.exists(preprocess_file):
            with open(preprocess_file, 'w') as f:
                f.write("asthma;:;disease\n")
                f.write("protein level;:;important\n")
            print(f"Created sample preprocessing file: {preprocess_file}")
        
        print("Test environment setup complete")

    def test_caching_ontology_from_url(self):
        # Test caching an ontology loaded from a URL
        print("Test caching an ontology loaded from a URL...")
        efo_cache = text2term.cache_ontology(ontology_url=self.EFO_URL, ontology_acronym="EFO", cache_folder=self.TEST_CACHE_FOLDER)
        print(f"Cache exists: {efo_cache.cache_exists()}\n")
        assert efo_cache.cache_exists() is True

        print("Test using the returned ontology cache object to map a list of terms...")
        mappings_efo_cache = efo_cache.map_terms(["asthma", "disease location", "food allergy"],
                                                 term_type=OntologyTermType.ANY)
        assert mappings_efo_cache.size > 0

    def test_caching_ontology_from_acronym(self):
        # Test caching an ontology by resolving its acronym using bioregistry
        print("Test caching an ontology by resolving its acronym using bioregistry...")
        clo_cache = text2term.cache_ontology(ontology_url="CLO", ontology_acronym="CLO", cache_folder=self.TEST_CACHE_FOLDER)
        print(f"Cache exists: {clo_cache.cache_exists()}\n")
        assert clo_cache.cache_exists() is True

    def test_caching_ontology_set(self):
        # Find ontologies.csv file in possible locations
        resources_paths = [
            os.path.join("text2term", "resources", "ontologies.csv"),
            os.path.join("..", "text2term", "resources", "ontologies.csv")
        ]
        
        ontology_registry_filepath = None
        for path in resources_paths:
            if os.path.exists(path):
                ontology_registry_filepath = path
                break
        
        if ontology_registry_filepath is None:
            self.skipTest("Ontology registry file not found in expected locations")
        
        nr_ontologies_in_registry = len(pd.read_csv(ontology_registry_filepath))

        # Test caching the set of ontologies specified in resources/ontologies.csv
        caches = text2term.cache_ontology_set(ontology_registry_filepath, cache_folder=self.TEST_CACHE_FOLDER)
        assert len(caches) == nr_ontologies_in_registry

    def test_mapping_to_cached_ontology(self):
        text2term.clear_cache(cache_folder=self.TEST_CACHE_FOLDER)
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test mapping a list of terms to EFO loaded from cache
        print("Test mapping a list of terms to EFO loaded from cache...")
        mappings_efo_cache = text2term.map_terms(["asthma", "disease location", "food allergy"], target_ontology="EFO",
                                                 use_cache=True, term_type=OntologyTermType.ANY, cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{mappings_efo_cache}\n")
        assert mappings_efo_cache.size > 0

        # Test mapping a list of terms to EFO loaded from a URL
        print("Test mapping a list of terms to EFO loaded from a URL...")
        mappings_efo_url = text2term.map_terms(["asthma", "disease location", "food allergy"],
                                               target_ontology=self.EFO_URL, term_type=OntologyTermType.ANY)
        print(f"{mappings_efo_url}\n")
        assert mappings_efo_url.size > 0

        # Test that mapping to cached ontology is the same as to ontology loaded from its URL
        print("Test that mapping to cached ontology is the same as to ontology loaded from its URL...")
        mappings_match = self.check_df_equals(self.drop_source_term_ids(mappings_efo_cache),
                                              self.drop_source_term_ids(mappings_efo_url))
        print(f"...{mappings_match}")
        assert mappings_match is True

    def test_mapping_to_cached_ontology_using_syntactic_mapper(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test mapping a list of terms to cached EFO using Jaro-Winkler syntactic similarity metric
        print("Test mapping a list of terms to cached ontology using Jaro-Winkler syntactic similarity metric...")
        df = text2term.map_terms(["asthma", "disease location", "food allergy"], "EFO", use_cache=True,
                                 mapper=text2term.Mapper.JARO_WINKLER, term_type=OntologyTermType.ANY, cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df}\n")
        assert df.size > 0

    def test_mapping_using_ontology_acronym(self):
        # Test mapping a list of terms by specifying the target ontology acronym, which gets resolved by bioregistry
        print(
            "Test mapping a list of terms to EFO by specifying the ontology acronym, which gets resolved by bioregistry")
        df2 = text2term.map_terms(["contains", "asthma"], "MONDO", term_type=OntologyTermType.CLASS)
        print(f"{df2}\n")
        assert df2.size > 0

    def test_mapping_tagged_terms(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test mapping a dictionary of tagged terms to cached EFO, and include unmapped terms in the output
        print("Test mapping a dictionary of tagged terms to cached EFO, and include unmapped terms in the output...")
        df3 = text2term.map_terms(
            {"asthma": "disease", "allergy": ["ignore", "response"], "protein level": ["measurement"],
             "isdjfnsdfwd": None}, target_ontology="EFO", excl_deprecated=True, use_cache=True, incl_unmapped=True,
            cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df3}\n")
        assert df3.size > 0
        assert df3[self.TAGS_COLUMN].str.contains("disease").any()
        assert df3[self.TAGS_COLUMN].str.contains("measurement").any()

    def test_preprocessing_from_file(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test processing tagged terms where the tags are provided in a file
        print("Test processing tagged terms where the tags are provided in a file...")
        
        # Ensure the file exists before trying to use it
        preprocess_file = "simple_preprocess.txt"
        if not os.path.exists(preprocess_file):
            with open(preprocess_file, 'w') as f:
                f.write("asthma;:;disease\n")
                f.write("protein level;:;important\n")

        tagged_terms = text2term.preprocess_tagged_terms(preprocess_file)
        df4 = text2term.map_terms(tagged_terms, target_ontology="EFO", use_cache=True, incl_unmapped=True,
                                  cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df4}\n")
        assert df4.size > 0
        assert df4[self.TAGS_COLUMN].str.contains("disease").any()
        assert df4[self.TAGS_COLUMN].str.contains("important").any()

    def test_preprocessing_tagged_terms(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        # Test processing tagged terms where the tags are provided in a file
        print("Test processing tagged terms where the tags are provided in a file...")

        # Ensure the file exists before trying to use it
        preprocess_file = "simple_preprocess.txt"
        if not os.path.exists(preprocess_file):
            with open(preprocess_file, 'w') as f:
                f.write("asthma;:;disease\n")
                f.write("protein level;:;important\n")

        tagged_terms = text2term.preprocess_tagged_terms(preprocess_file, template_path="test_templates.txt")
        df = text2term.map_terms(tagged_terms, target_ontology="EFO", use_cache=True, incl_unmapped=True,
                                 cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df}\n")
        assert df.size > 0
        assert df[self.TAGS_COLUMN].str.contains("disease").any()
        assert df[self.TAGS_COLUMN].str.contains("important").any()
        assert df["Source Term"].str.contains("hypertension").any()
        assert not df["Source Term"].str.contains("hypertension NOS").all()

    def test_preprocessing_terms(self):
        input_terms = ['Hypertension NOS', 'Diabetes mellitus due to underlying condition']
        result = text2term.preprocess_terms(terms=input_terms, template_path='test_templates.txt')
        expected = {
            'Hypertension NOS': 'Hypertension',
            'Diabetes mellitus due to underlying condition': 'Diabetes mellitus'
        }
        self.assertEqual(result, expected)

    def test_preprocessing_terms_and_dedupe(self):
        input_terms = ['Hypertension NOS', 'Diabetes mellitus due to underlying condition',
                       'Hypertension due to unspecified condition', 'Hypertension']
        result = text2term.preprocess_terms(terms=input_terms, template_path='test_templates.txt', rem_duplicates=True)
        expected = {
            'Hypertension': 'Hypertension',
            'Diabetes mellitus due to underlying condition': 'Diabetes mellitus'
        }
        self.assertEqual(result, expected)

    def test_preprocessing_blocklisted_terms(self):
        terms = ['Hypertension', 'Patient ID', 'Admission Date']
        blocklist_path = 'test_blocklist.txt'  # This file should contain 'Common Cold'
        result = text2term.preprocess_terms(terms, template_path="", blocklist_path=blocklist_path)
        expected = {
            'Hypertension': 'Hypertension'
        }
        self.assertEqual(result, expected)

    def test_mapping_to_properties(self):
        # Test mapping a list of properties to EFO loaded from a URL and restrict search to properties
        print("Test mapping a list of properties to EFO loaded from a URL and restrict search to properties...")
        df5 = text2term.map_terms(source_terms=["contains", "location"], target_ontology=self.EFO_URL,
                                  term_type=OntologyTermType.PROPERTY)
        print(f"{df5}\n")
        assert df5.size > 0

        # Test mapping a list of properties to EFO loaded from cache and restrict search to properties
        print("Test mapping a list of properties to EFO loaded from cache and restrict search to properties...")
        self.ensure_cache_exists("EFO", self.EFO_URL)
        df6 = text2term.map_terms(source_terms=["contains", "location"], target_ontology="EFO", use_cache=True,
                                  term_type=OntologyTermType.PROPERTY, cache_folder=self.TEST_CACHE_FOLDER)
        print(f"{df6}\n")
        assert df6.size > 0

        # Test that mapping to properties in cached ontology is the same as to ontology loaded from its URL
        properties_df_match = self.check_df_equals(self.drop_source_term_ids(df5), self.drop_source_term_ids(df6))
        print(f"Properties match: {properties_df_match}")
        assert properties_df_match is True

    def test_mapping_zooma_ontologies(self):
        # Test mapping a list of terms to multiple ontologies using the Zooma mapper
        print("Test mapping a list of terms to multiple ontologies using the Zooma mapper...")
        df_zooma = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="EFO,NCIT",
                                       mapper=Mapper.ZOOMA, term_type=OntologyTermType.ANY)
        print(f"{df_zooma}\n")
        assert df_zooma.size > 0
        assert df_zooma[self.MAPPED_TERM_CURIE_COLUMN].str.contains("EFO:").any()
        assert df_zooma[self.MAPPED_TERM_CURIE_COLUMN].str.contains("NCIT:").any()

    def test_mapping_zooma_empty_response(self):
        print("Test mapping a term that obviously has no ontology term to be mapped to...")
        df_zooma = text2term.map_terms(["ziggy"], target_ontology="NCIT",
                                       mapper=Mapper.ZOOMA, term_type=OntologyTermType.ANY)
        print(f"{df_zooma}\n")
        assert df_zooma.empty is True

    def test_mapping_zooma_bad_url(self):
        print("Test mapping terms to a wrong URL for Zooma mapper...")
        bp_mapper = ZoomaMapper()
        with self.assertRaises(Exception):
            bp_mapper._do_get_request(request_url="http://www.ebi.ac.uk/spot/zooma/v2/api/services/annotateBad")

    def test_mapping_bioportal_ontologies_no_apikey(self):
        # Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper without API Key
        print("Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper...")
        df_bioportal = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="EFO,NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY)
        assert df_bioportal.empty is True

    def test_mapping_bioportal_ontologies(self):
        # Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper
        print("Test mapping a list of terms to multiple ontologies using the BioPortal Annotator mapper...")
        df_bioportal = text2term.map_terms(["asthma", "location", "food allergy"], target_ontology="EFO,NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY,
                                           bioportal_apikey="8f0cbe43-2906-431a-9572-8600d3f4266e")
        print(f"{df_bioportal}\n")
        assert df_bioportal.size > 0
        assert df_bioportal[self.MAPPED_TERM_CURIE_COLUMN].str.contains("EFO:").any()
        assert df_bioportal[self.MAPPED_TERM_CURIE_COLUMN].str.contains("NCIT:").any()

    def test_mapping_bioportal_empty_response(self):
        print("Test mapping a term that obviously has no ontology term to be mapped to...")
        df_bioportal = text2term.map_terms(["ziggy"], target_ontology="NCIT",
                                           mapper=Mapper.BIOPORTAL, term_type=OntologyTermType.ANY,
                                           bioportal_apikey="8f0cbe43-2906-431a-9572-8600d3f4266e")
        print(f"{df_bioportal}\n")
        assert df_bioportal.empty is True

    def test_mapping_bioportal_bad_url(self):
        print("Test mapping terms to a wrong URL for BioPortal Annotator mapper...")
        bp_mapper = BioPortalAnnotatorMapper(bp_api_key="8f0cbe43-2906-431a-9572-8600d3f4266e")
        with self.assertRaises(Exception):
            bp_mapper._do_get_request(request_url="http://data.bioontology.org/annotatorBad")

    def test_term_collector(self):
        expected_nr_efo_terms = 50867
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms()
        assert len(terms) == expected_nr_efo_terms

    def test_term_collector_classes_only(self):
        expected_nr_efo_classes = 50643
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(term_type=OntologyTermType.CLASS)
        assert len(terms) == expected_nr_efo_classes

    def test_term_collector_properties_only(self):
        expected_nr_efo_properties = 224
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(term_type=OntologyTermType.PROPERTY)
        assert len(terms) == expected_nr_efo_properties

    def test_term_collector_iri_limit(self):
        efo_base_iri = "http://www.ebi.ac.uk/efo/"
        expected_nr_terms_with_efo_iri = 17383
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(base_iris=[efo_base_iri], term_type=OntologyTermType.ANY)
        assert len(terms) == expected_nr_terms_with_efo_iri

    def test_term_collector_iri_limit_properties_only(self):
        efo_base_iri = "http://www.ebi.ac.uk/efo/"
        expected_nr_properties_with_efo_iri = 29
        efo_term_collector = OntologyTermCollector(ontology_iri=self.EFO_URL)
        terms = efo_term_collector.get_ontology_terms(base_iris=[efo_base_iri], term_type=OntologyTermType.PROPERTY)
        assert len(terms) == expected_nr_properties_with_efo_iri

    def test_mapping_with_min_score_filter(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        min_score = 0.6
        search_terms = ["asthma attack", "location"]

        print("Test mapping to cached EFO using Zooma mapper and min_score filter...")
        df_zooma = text2term.map_terms(search_terms, target_ontology="EFO,NCIT", mapper=Mapper.ZOOMA,
                                       term_type=OntologyTermType.ANY, min_score=min_score)
        assert (df_zooma[self.MAPPING_SCORE_COLUMN] >= min_score).all()

        print("Test mapping to cached EFO using TFIDF similarity metric and min_score filter...")
        df_tfidf = text2term.map_terms(search_terms, target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                       term_type=OntologyTermType.ANY, min_score=min_score, cache_folder=self.TEST_CACHE_FOLDER)
        assert (df_tfidf[self.MAPPING_SCORE_COLUMN] >= min_score).all()

        print("Test mapping to cached EFO using Levenshtein similarity metric and min_score filter...")
        df_leven = text2term.map_terms(search_terms, target_ontology="EFO", use_cache=True, mapper=Mapper.LEVENSHTEIN,
                                       term_type=OntologyTermType.ANY, min_score=min_score, cache_folder=self.TEST_CACHE_FOLDER)
        assert (df_leven[self.MAPPING_SCORE_COLUMN] >= min_score).all()

    def test_mapping_with_min_score_filter_empty_results(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        print("Test mapping to EFO using TFIDF similarity metric and min_score filter that results in no mappings...")
        df_tfidf = text2term.map_terms(["carbon monoxide"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                       term_type=OntologyTermType.ANY, min_score=0.99, cache_folder=self.TEST_CACHE_FOLDER)
        assert df_tfidf.empty is True

    def test_include_unmapped_terms(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        df = text2term.map_terms(["asthma", "margarita"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                 incl_unmapped=True, min_score=0.8, cache_folder=self.TEST_CACHE_FOLDER)
        assert df[self.TAGS_COLUMN].str.contains("unmapped").any()

    def test_exclude_metadata_from_output_file(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        test_output_file = "test_output_without_metadata.csv"
        text2term.map_terms(["asthma"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                 excl_metadata=True, save_mappings=True, output_file=test_output_file, cache_folder=self.TEST_CACHE_FOLDER)
        with open(test_output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            assert not first_line.startswith('#'), "CSV output file should not start with metadata header"

    def test_include_metadata_in_output_file(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        test_output_file = "test_output_with_metadata.csv"
        text2term.map_terms(["asthma"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                 excl_metadata=False, save_mappings=True, output_file=test_output_file, cache_folder=self.TEST_CACHE_FOLDER)
        with open(test_output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            assert first_line.startswith('#'), "CSV output file should contain metadata header"

    def test_include_unmapped_terms_when_mappings_df_is_empty(self):
        self.ensure_cache_exists("EFO", self.EFO_URL)
        df = text2term.map_terms(["mojito", "margarita"], target_ontology="EFO", use_cache=True, mapper=Mapper.TFIDF,
                                 incl_unmapped=True, min_score=0.8, cache_folder=self.TEST_CACHE_FOLDER)
        assert df[self.TAGS_COLUMN].str.contains("unmapped").any()

    def drop_source_term_ids(self, df):
        # Unless specified, source term IDs are randomly generated UUIDs. We have to drop the ID column to be able to
        # get a meaningful diff between two dataframes. Otherwise, the dataframes would always differ because of the IDs
        return df.drop(self.SOURCE_TERM_ID_COLUMN, axis=1)

    def check_df_equals(self, df, expected_df):
        # Use pandas::assert_frame_equal function to determine if two data frames are equal
        pd.testing.assert_frame_equal(df, expected_df, check_names=False, check_like=True)
        return True

    def ensure_cache_exists(self, ontology_name, ontology_url):
        if not text2term.cache_exists(ontology_name, cache_folder=self.TEST_CACHE_FOLDER):
            text2term.cache_ontology(ontology_url=ontology_url, ontology_acronym=ontology_name, cache_folder=self.TEST_CACHE_FOLDER)


class OntoUtilsTestSuite(unittest.TestCase):

    def test_normalize_list(self):
        self.assertEqual(onto_utils.normalize_list([" Apple ", "Banana"]), ["apple", "banana"])

    def test_normalize(self):
        self.assertEqual(onto_utils.normalize(" TeSt "), "test")

    def test_remove_quotes(self):
        self.assertEqual(onto_utils.remove_quotes('"quoted"'), "quoted")
        self.assertEqual(onto_utils.remove_quotes("'quoted'"), "quoted")

    def test_remove_whitespace(self):
        self.assertEqual(onto_utils.remove_whitespace(" spaced string "), "spacedstring")

    def test_curie_from_iri(self):
        self.assertEqual(onto_utils.curie_from_iri("http://purl.obolibrary.org/obo/GO_0008150"), "GO:0008150")

    def test_label_from_iri(self):
        self.assertEqual(onto_utils.label_from_iri("http://example.org/ontology/Term_Label"), "Term_Label")

    def test_iri_from_tag(self):
        self.assertEqual(onto_utils.iri_from_tag("GO_0008150"), "http://purl.obolibrary.org/obo/GO_0008150")
        self.assertEqual(onto_utils.iri_from_tag("GO:0008150"), "http://purl.obolibrary.org/obo/GO_0008150")

    def test_get_logger(self):
        logger = onto_utils.get_logger("test_logger")
        self.assertTrue(hasattr(logger, "info"))

    def test_parse_list_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("apple\nbanana\n")
            f_path = f.name
        try:
            self.assertEqual(onto_utils.parse_list_file(f_path), ["apple", "banana"])
        finally:
            os.remove(f_path)

    def test_parse_csv_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("disease,disease code,patient count\n"
                    "asthma,abb1,30\n"
                    "bronchitis,baa1,21")
            f_path = f.name
        try:
            terms, term_ids = onto_utils.parse_csv_file(file_path=f_path,
                                                        term_column_name="disease",
                                                        term_id_column_name="disease code")
            self.assertEqual(terms.tolist(), ["asthma", "bronchitis"])
            self.assertEqual(term_ids.tolist(), ["abb1", "baa1"])
        finally:
            os.remove(f_path)

    def test_parse_tsv_file(self):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            f.write("disease\tdisease code\tpatient count\n"
                    "asthma\tabb1\t30\n"
                    "bronchitis\tbaa1\t21")
            f_path = f.name
        try:
            terms, term_ids = onto_utils.parse_tsv_file(file_path=f_path,
                                                        term_column_name= "disease",
                                                        term_id_column_name= "disease code")
            self.assertEqual(terms.tolist(), ["asthma", "bronchitis"])
            self.assertEqual(term_ids.tolist(), ["abb1", "baa1"])
        finally:
            os.remove(f_path)

    def test_get_ontology_from_labels(self):
        ontology_iri = "https://test.ontology.org/"
        terms = ["asthma", "bronchitis", "hypertension", "diabetes mellitus"]
        result = onto_utils.get_ontology_from_labels(terms, ontology_iri=ontology_iri)
        self.assertEqual(result.base_iri, ontology_iri)
        self.assertEqual(len(list(result.classes())), len(terms))

    def test_generate_uuid(self):
        some_uuid = onto_utils.generate_uuid(length=23)
        self.assertTrue(len(some_uuid) == 23)

    def test_generate_iri(self):
        iri = onto_utils.generate_iri()
        self.assertTrue(iri.startswith(onto_utils.BASE_IRI + "R"))

    def test_generate_iris(self):
        iris = onto_utils.generate_iris(3)
        self.assertEqual(len(iris), 3)
        self.assertTrue(all(i.startswith(onto_utils.BASE_IRI + "R") for i in iris))


if __name__ == '__main__':
    unittest.main()
