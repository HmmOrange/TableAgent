__all__ = [
    "FormulaRelationOperator",
    "MultiTableOperator",
    "RelationalTableOperator",
    "TableOperators",
    "TableRoutingOperator",
]

def __getattr__(name):
    if name == "TableOperators":
        from TableAgent.QA.operators.table_operator import TableOperators
        return TableOperators
    if name == "MultiTableOperator":
        from TableAgent.QA.operators.multitab_operator import MultiTableOperator
        return MultiTableOperator
    if name == "TableRoutingOperator":
        from TableAgent.QA.operators.table_routing_operator import TableRoutingOperator
        return TableRoutingOperator
    if name == "RelationalTableOperator":
        from TableAgent.QA.operators.relational_table_operator import RelationalTableOperator
        return RelationalTableOperator
    if name == "FormulaRelationOperator":
        from TableAgent.QA.operators.formula_relation_operator import FormulaRelationOperator
        return FormulaRelationOperator
    raise AttributeError(name)
