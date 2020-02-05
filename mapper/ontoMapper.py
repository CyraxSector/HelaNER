from mapper.ontoLoader import ontoLoader


def main():
    model_path = 'E:\\Projects\\HelaNER\\data\\hela-ontology.owl'
    mapper = ontoLoader()
    mapper.load_kb(model_path)


if __name__ == "__main__":
    main()
