x = 1
y = {
    "key": "value with (parens) and = sign",
    "list": [
        {"a": 1, "b": "string with (paren", "c": "string with close paren) and (more)"},
        {"a": 2},
    ],
    "another": "another string",
    "step": {
        "step_id": "12.1",
        "primary_document": {
            "node_id": "ABC/X/Y/Z",
            "title": "Some Title",
            "evidence": "ANS.foo (да — Bar=Baz) + audit — не найдено, "
                        "вывод по аналогии"},
        },
        "required_nsi": [
            {"node_id": "XYZ/Constants/RegFlag",
             "create_from_form": False,
             "evidence": "graph_gap (constraint:220 — нет бух. контура)"},
        ],
    },
}
