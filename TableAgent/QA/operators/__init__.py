__all__ = ["TableOperators"]

def __getattr__(name):
    if name == "TableOperators":
        from TableAgent.QA.operators.table_operator import TableOperators
        return TableOperators
    raise AttributeError(name)
