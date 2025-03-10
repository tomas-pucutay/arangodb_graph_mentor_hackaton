def get_competences(db):
    aql = """
        FOR comp in Competence
        RETURN comp.node_name
    """

    cursor = db.aql.execute(aql)
    response = list(cursor)
    return response