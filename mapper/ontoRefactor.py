from mapper.ontoUtils import KnowledgeBase, KBEntity, KBRelation
from lxml import etree


class KBLoader(object):
    @staticmethod
    def import_owl_kb(kb_name, kb_filename):
        """
        Create a KnowledgeBase object with entities and relations from an OWL file
        :param kb_name:
        :param kb_filename:
        :return:
        """

        # get the description label for this resource id
        def get_label(l):
            if l.text is not None:
                return l.text
            else:
                r_id = l.get('{' + ns['rdf'] + '}resource')
                if r_id in descriptions:
                    return descriptions[r_id][0]
            return None

        assert kb_filename.endswith('.owl') or kb_filename.endswith('.rdf')

        # initialize the KB
        kb = KnowledgeBase()
        kb.name = kb_name

        # parse the file
        try:
            tree = etree.parse(kb_filename)
        except etree.XMLSyntaxError:
            p = etree.XMLParser(huge_tree=True)
            tree = etree.parse(kb_filename, parser=p)

        root = tree.getroot()
        ns = root.nsmap

        if None in ns:
            del ns[None]

        # get description dict
        descriptions = dict()
        for desc in root.findall('rdf:Description', ns):
            resource_id = desc.get('{' + ns['rdf'] + '}about')
            try:
                labels = []
                for label in desc.findall('rdfs:label', ns):
                    if label.text is not None:
                        labels.append(label.text)
                if 'skos' in ns:
                    for label in desc.findall('skos:prefLabel', ns):
                        if label.text is not None:
                            labels.append(label.text)
                if 'oboInOwl' in ns:
                    for syn in desc.findall('oboInOwl:hasExactSynonym', ns):
                        if syn.text is not None:
                            labels.append(syn.text)
                    for syn in desc.findall('oboInOwl:hasRelatedSynonym', ns) \
                               + desc.findall('oboInOwl:hasNarrowSynonym', ns) \
                               + desc.findall('oboInOwl:hasBroadSynonym', ns):
                        if syn.text is not None:
                            labels.append(syn.text)
                if len(labels) > 0:
                    descriptions[resource_id] = labels
            except AttributeError:
                continue

        # parse OWL classes
        for cl in root.findall('owl:Class', ns):
            # instantiate an entity.
            research_entity_id = cl.get('{' + ns['rdf'] + '}about')
            entity = KBEntity(research_entity_id, None, [], '')

            # list of KBRelations to add
            relations = []

            if entity.research_entity_id is not None and entity.research_entity_id != '':
                try:
                    labels = []

                    # get rdfs labels
                    for label in cl.findall('rdfs:label', ns):
                        l_text = get_label(label)
                        if l_text is not None:
                            labels.append(l_text)

                    # add labels from description
                    if entity.research_entity_id in descriptions:
                        labels += descriptions[entity.research_entity_id]

                    # get skos labels
                    if 'skos' in ns:
                        for label in cl.findall('skos:prefLabel', ns):
                            l_text = get_label(label)
                            if l_text is not None:
                                labels.append(l_text)
                        for label in cl.findall('skos:altLabel', ns):
                            l_text = get_label(label)
                            if l_text is not None:
                                labels.append(l_text)
                        for label in cl.findall('skos:hiddenLabel', ns):
                            l_text = get_label(label)
                            if l_text is not None:
                                labels.append(l_text)

                    # get synonyms
                    if 'oboInOwl' in ns:
                        for syn in cl.findall('oboInOwl:hasExactSynonym', ns):
                            l_text = get_label(syn)
                            if l_text is not None:
                                labels.append(l_text)
                        for syn in cl.findall('oboInOwl:hasRelatedSynonym', ns) \
                                   + cl.findall('oboInOwl:hasNarrowSynonym', ns) \
                                   + cl.findall('oboInOwl:hasBroadSynonym', ns):
                            l_text = get_label(syn)
                            if l_text is not None:
                                labels.append(l_text)

                    # set canonical_name and aliases
                    if len(labels) > 0:
                        entity.canonical_name = labels[0]
                        entity.aliases = list(
                            set([lab.lower() for lab in labels])
                        )

                    # if no name available (usually entity from external KB), replace name with id
                    if entity.canonical_name is None:
                        entity.canonical_name = entity.research_entity_id

                    # get definition
                    if 'skos' in ns:
                        for definition in cl.findall('skos:definition', ns):
                            if definition.text is not None:
                                entity.definition += definition.text.lower(
                                ) + ' '
                    if 'obo' in ns:
                        for definition in cl.findall('obo:IAO_0000115', ns):
                            if definition.text is not None:
                                entity.definition += definition.text.lower(
                                ) + ' '
                    entity.definition = entity.definition.strip()

                    # get subclass relations
                    for sc_rel in cl.findall('rdfs:subClassOf', ns):
                        target_research_entity_id = sc_rel.get('{' + ns['rdf'] + '}resource', ns)
                        if isinstance(target_research_entity_id, str):
                            relation = KBRelation(
                                relation_type='subClassOf',
                                entity_ids=[
                                    entity.research_entity_id,
                                    target_research_entity_id
                                ],
                                symmetric=False
                            )
                            relations.append(relation)
                except AttributeError:
                    pass

                # add relations to entity and to kb
                for rel in relations:
                    kb.add_relation(rel)
                    rel_index = len(kb.relations) - 1
                    entity.relation_ids.append(rel_index)

                # add entity to kb
                kb.add_entity(entity)

        return kb
