import os
import sys

import requests
from pprint import pprint
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator

from mapper.ontoUtils import KnowledgeBase
from mapper.ontoRefactor import KBLoader


class ontoLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_kb(kb_path) -> KnowledgeBase:
        """
        Load KnowledgeBase specified at kb_path
        :param kb_path: path to knowledge base
        :return:
        """
        sys.stdout.write("\tLoading %s...\n" % kb_path)

        assert kb_path is not None
        assert kb_path != ''

        kb_name = os.path.basename(kb_path)

        # load kb
        if kb_path.endswith('.owl') or kb_path.endswith('.rdf') or \
                kb_path.endswith('.OWL') or kb_path.endswith('.RDF'):
            kb = KBLoader.import_owl_kb(kb_name, kb_path)
        elif kb_path.endswith('.ttl') or kb_path.endswith('.n3'):
            sys.stdout.write('This program cannot parse your file type.\n')
            raise NotImplementedError()
        else:
            val = URLValidator()
            try:
                val(kb_path)
            except ValidationError:
                raise

            response = requests.get(kb_path, stream=True)
            response.raise_for_status()
            temp_file = 'temp_file_ontoemma.owl'
            with open(temp_file, 'wb') as outf:
                for block in response.iter_content(1024):
                    outf.write(block)
            kb = KBLoader.import_owl_kb('', temp_file)
            os.remove(temp_file)

        sys.stdout.write("\tEntities: %i\n" % len(kb.entities))
        pprint(kb.entities)

        return kb
