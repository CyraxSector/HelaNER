from owlready2 import World, re

from ontology.phrases import comparison_question_phrase


class Comparison:
    def __init__(self):
        my_world = World()
        my_world.get_ontology("file://E:\\Projects\\HelaNER\\data\\hela-ontology.owl").load()
        self.graph = my_world.as_rdflib_graph()

    def comparison_question(self, class_from_text):
        query = """PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                    SELECT ?subject ?supertype
                                    WHERE{
                                    ?subject rdfs:subClassOf ?supertype.
                                    FILTER (regex(str(?supertype), '""" + class_from_text + """'))
                                    }"""

        results_list = self.graph.query(query)
        selected_results = []
        count = 0

        for item in results_list:
            super_class = str(item['supertype'].toPython())
            super_class = re.sub(r'.*#', "", super_class)

            sub_class = str(item['subject'].toPython())
            sub_class = re.sub(r'.*#', "", sub_class)

            selected_results.append(
                {super_class: sub_class})  # create a dictionary
            count = count + 1

        sub_list = []
        if count > 1:
            for element in selected_results:
                for val in element.values():
                    sub_list.append(val)
            return sub_list

    def comparison_answer(self, sub_classes):
        question = ''
        length = len(sub_classes)
        for i in range(length):
            question = question + sub_classes[i] + ' ,'
        question = question + comparison_question_phrase[0]
        print(question)

        final_answer = ''

        for element in sub_classes:
            query = """PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                SELECT ?domain ?range ?property 
                                WHERE{
                                ?property rdfs:domain ?domain;
                                    rdfs:range ?range.
                                FILTER (regex(str(?domain), '""" + element + """'))
                                }"""

            results_list = self.graph.query(query)  # query result will be stored in a variable
            selected_results = []
            for item in results_list:
                predicate = str(item['property'].toPython())
                predicate = re.sub(r'.*#', "", predicate)

                domain = str(item['domain'].toPython())
                domain = re.sub(r'.*#', "", domain)

                range_class = str(item['range'].toPython())
                range_class = re.sub(r'.*#', "", range_class)

                selected_results.append(
                    {'subject': domain, 'object': range_class, 'predicate': predicate})  # create a dictionary

                for result in selected_results:
                    if result['subject'] == element:
                        answer = result['subject'] + ' ' + result['object'] + ' ' + result['predicate']

                final_answer = final_answer + answer + '\n'

        print(final_answer)
        return final_answer
