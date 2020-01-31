from ontology.toupleAnalyzer import create_tuples


def send_data_to_onto(sov):
    for element in sov:
        if len(element) != 0:
            subject_of_sentence = element[0]
            object_of_sentence = element[1]
            verb_of_sentence = element[2]
            create_tuples(subject_of_sentence, object_of_sentence, verb_of_sentence)
