from ontology.ontoAnalyzer import load_ontology
from ontology.queryAnalyzer import Comparison
from ontology.ontoProcessor import extract_sov, tagging
from ontology.preProcessor import cleaning
from ontology.toupleProcessor import send_data_to_onto

pre_processed_text = cleaning()
sov_from_text = []


def call_tagging(cleaned_text):
    fully_tagged = []
    for sent in cleaned_text:
        tagged_text = tagging(sent)
        fully_tagged.append(tagged_text)
    print(fully_tagged)
    return fully_tagged


def call_extract_sov(tagged_text):
    for sent in tagged_text:
        extracted_triples_from_text = extract_sov(sent)
        sov_from_text.append(extracted_triples_from_text)
    print(sov_from_text)
    return sov_from_text


def available_classes(sov_list):
    classes = []
    class_list = []
    for element in sov_list:
        if len(element) != 0:
            classes.append(element[0])
            classes.append(element[1])
    class_set = set(classes)
    class_list.extend(class_set)
    print(class_list)
    return class_list


def call_comparison_qa(classes):
    run_query = Comparison()
    for element in classes:
        sub_classes = run_query.comparison_question(element)
        if sub_classes:
            run_query.comparison_answer(sub_classes)


text_tagged = call_tagging(pre_processed_text)
sov = call_extract_sov(text_tagged)
send_data_to_onto(sov)
classes_in_text = available_classes(sov)
load_ontology(classes_in_text)
