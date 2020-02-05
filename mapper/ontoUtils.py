import json
from collections import defaultdict


class KBEntity(object):
    def __init__(
            self,
            research_entity_id=None,
            canonical_name=None,
            aliases=[],
            definition="<s>",
            source_urls=[],
            category=None
    ):
        self.research_entity_id = research_entity_id
        self.canonical_name = canonical_name
        self.aliases = aliases
        self.definition = definition
        self.source_urls = source_urls
        self.category = category
        # relations is a list of KBRelation ids.
        self.relation_ids = []
        self.other_contexts = []
        self.additional_details = defaultdict(list)

        # Fields containing tokenized text used for training
        self.tokenized_definition = None
        self.tokenized_canonical_name = None
        self.tokenized_aliases = None

    def __repr__(self):
        return json.dumps(
            {
                'research_entity_id': self.research_entity_id,
                'canonical_name': self.canonical_name
            }
        )

    def __eq__(self, ent):
        if self.research_entity_id == ent.research_entity_id \
                and self.canonical_name == ent.canonical_name \
                and set(self.aliases) == set(ent.aliases) \
                and self.source_urls == ent.source_urls \
                and self.category == ent.category:
            return True
        else:
            return False

    @property
    def raw_ids(self):
        return self.research_entity_id.split('|')


class KBRelation(object):
    def __init__(self, relation_type=None, entity_ids=None, symmetric=None, labels=None):
        self.relation_type = relation_type
        self.entity_ids = entity_ids
        self.labels = set() if labels is None else labels
        self.symmetric = symmetric


class KnowledgeBase(object):
    def __init__(self):
        self.name = ""
        self.entities = []
        self.relations = []
        self.research_entity_id_to_entity_index = dict()
        self.raw_id_to_entity_index = dict()
        self.canonical_name_to_entity_index = defaultdict(set)
        self.entity_ids_to_relation_index = defaultdict(set)
        self.null_entity = None

    def add_entity(self, new_entity: KBEntity):
        """
        Verify and add entity to the KB
        :param new_entity: A new entity object to be added to the KB
        :return:
        """
        if self.validate_entity(new_entity):
            self.entities.append(new_entity)
            ent_index = len(self.entities) - 1
            self.research_entity_id_to_entity_index[
                new_entity.research_entity_id
            ] = ent_index
            for raw_id in new_entity.raw_ids:
                self.raw_id_to_entity_index[raw_id] = ent_index
            self.canonical_name_to_entity_index[new_entity.canonical_name
            ].add(ent_index)
        else:
            raise ValueError('Entity failed validation: %s' % new_entity)
        return

    @staticmethod
    def validate_entity(ent: KBEntity):
        """
        Check if input entity is valid
        :param ent:
        :return: bool
        """
        if ent.canonical_name is None \
                or ent.canonical_name == "" \
                or ent.research_entity_id is None \
                or ent.research_entity_id == "":
            return False
        else:
            return True

    @staticmethod
    def validate_relation(rel: KBRelation):
        """
        Check if input relation is valid
        :param rel:
        :return: bool
        """
        if rel.relation_type is None \
                or rel.relation_type == "" \
                or rel.entity_ids[0] is None \
                or rel.entity_ids[1] is None:
            return False
        else:
            return True

    def add_relation(self, new_relation: KBRelation):
        """
        Verify and add relation to KB
        :param new_relation:
        :return:
        """
        if self.validate_relation(new_relation):
            self.relations.append(new_relation)
            rel_index = len(self.relations) - 1
            self.entity_ids_to_relation_index[tuple(new_relation.entity_ids)].add(rel_index)
        else:
            raise ValueError('Relation failed validation: %s' % new_relation)
        return
