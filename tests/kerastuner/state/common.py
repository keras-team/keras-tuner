import json


def exportable_attributes_exists(state):
    for attribute in state.exportable_attributes:
        getattr(state, attribute)


def is_serializable(state):
    conf = state.to_dict()
    # serialize and deserialize
    json_conf = json.loads(json.dumps(conf))
    assert conf == json_conf
