import fastkml as kml


def print_child_features(element):
    if not getattr(element, 'features', None):
        return
    for feature in element.features():
        print(feature.name)
        print_child_features(feature)


fname = "171440.kml"

k = kml.KML()

with open(fname, encoding='UTF-8') as kmlFile:
    k.from_string(kmlFile.read().encode('utf-8'))

print_child_features(k)
