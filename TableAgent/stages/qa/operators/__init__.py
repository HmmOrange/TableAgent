__all__ = [
    "FormulaRelationOperator",
    "MultiTableOperator",
    "RelationalTableOperator",
    "TableOperators",
    "TableRoutingOperator",
]

def __getattr__(name):
    if name == "TableOperators":
        from TableAgent.stages.qa.operators.table_operator import TableOperators
        return TableOperators
    if name == "MultiTableOperator":
        from TableAgent.stages.qa.operators.multitab_operator import MultiTableOperator
        return MultiTableOperator
    if name == "TableRoutingOperator":
        from TableAgent.stages.qa.operators.table_routing_operator import TableRoutingOperator
        return TableRoutingOperator
    if name == "RelationalTableOperator":
        from TableAgent.stages.qa.operators.relational_table_operator import RelationalTableOperator
        return RelationalTableOperator
    if name == "FormulaRelationOperator":
        from TableAgent.stages.qa.operators.formula_relation_operator import FormulaRelationOperator
        return FormulaRelationOperator
    raise AttributeError(name)
