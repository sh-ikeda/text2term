import argparse
import os
from t2t import map_terms, cache_ontology
from onto_cache import cache_exists
from mapper import Mapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A tool for mapping free-text descriptions of (biomedical) '
                                                 'entities to ontology terms')
    parser.add_argument("-s", "--source", required=True, type=str,
                        help="Input file containing 'source' terms to map to ontology terms: list of terms or CSV file")
    parser.add_argument("-t", "--target", required=True, type=str,
                        help="Path or URL of 'target' ontology to map source terms to. When the chosen mapper is "
                             "BioPortal or Zooma, provide a comma-separated list of acronyms (eg 'EFO,HPO') or write "
                             "'all' to search all ontologies")
    parser.add_argument("-o", "--output", required=False, type=str, default="",
                        help="Path to desired output file for the mappings (default=current working directory)")
    parser.add_argument("-m", "--mapper", required=False, type=str, default="tfidf",
                        help="Method used to compare source terms with ontology terms. One of: " + str(Mapper.list()) +
                             " (default=tfidf)")
    parser.add_argument("-csv", "--csv_input", required=False, type=str, default=(),
                        help="Specifies that the input is a CSV file—This should be followed by the name of the column "
                             "that contains the terms to map, optionally followed by the name of the column that "
                             "contains identifiers for the terms (eg 'my_terms,my_term_ids')")
    parser.add_argument("-sep", "--separator", required=False, type=str, default=',',
                        help="Specifies the cell separator to be used when reading a table")
    parser.add_argument("-top", "--top_mappings", required=False, type=int, default=3,
                        help="Maximum number of top-ranked mappings returned per source term (default=3)")
    parser.add_argument("-min", "--min_score", required=False, type=float, default=0.5,
                        help="Minimum similarity score [0,1] for the mappings (1=exact match; default=0.5)")
    parser.add_argument("-iris", "--base_iris", required=False, type=str, default=(),
                        help="Map only to ontology terms whose IRIs start with a value given in this comma-separated "
                             "list (eg 'http://www.ebi.ac.uk/efo,http://purl.obolibrary.org/obo/HP)')")
    parser.add_argument("-d", "--excl_deprecated", required=False, default=False, action="store_true",
                        help="Exclude ontology terms stated as deprecated via `owl:deprecated true` (default=False)")
    parser.add_argument("-g", "--save_term_graphs", required=False, default=False, action="store_true",
                        help="Save vis.js graphs representing the neighborhood of each ontology term (default=False)")
    parser.add_argument("-c", "--store_in_cache", required=False, type=str, default="",
                        help="Cache the target ontology using the name given here")
    parser.add_argument("-type", "--term_type", required=False, type=str, default="class",
                        help="Define whether to map to ontology classes, properties, or both")
    parser.add_argument('-u', "--incl_unmapped", required=False, default=False, action="store_true",
                        help="Include all unmapped terms in the output")
    parser.add_argument('-bp', "--bioportal_apikey", required=False, type=str, default="",
                        help="BioPortal API Key to use along with the BioPortal mapper option")
    parser.add_argument('-md', "--excl_metadata", required=False, default=False, action="store_true",
                        help="Exclude metadata in the output file")
    parser.add_argument("-k", "--keep_sep_char", required=False, default=False, action="store_true",
                        help="Do not replace non-alphanumeric characters with space before tokenization")

    arguments = parser.parse_args()
    if not os.path.exists(arguments.source):
        parser.error("The file '{}' does not exist".format(arguments.source))
    mapper = Mapper(arguments.mapper)
    iris = arguments.base_iris
    if len(iris) > 0:
        iris = tuple(iris.split(','))
    csv_columns = arguments.csv_input
    if len(csv_columns) > 0:
        csv_columns = tuple(csv_columns.split(','))
    target = arguments.target
    acronym = arguments.store_in_cache
    if acronym != "":
        cache_ontology(target, acronym, iris)
        target = acronym
    map_terms(arguments.source, target, output_file=arguments.output, csv_columns=csv_columns,
              excl_deprecated=arguments.excl_deprecated, mapper=mapper, max_mappings=arguments.top_mappings,
              min_score=arguments.min_score, base_iris=iris, save_graphs=arguments.save_term_graphs,
              save_mappings=True, separator=arguments.separator, use_cache=cache_exists(target),
              term_type=arguments.term_type, incl_unmapped=arguments.incl_unmapped, excl_metadata=arguments.excl_metadata,
              bioportal_apikey=arguments.bioportal_apikey, keep_sep_char=arguments.keep_sep_char)
