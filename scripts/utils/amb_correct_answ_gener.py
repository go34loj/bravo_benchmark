import sqlite3

DB_PATH = "../unified_database.db"


# ---------- Mapping ----------

SUBCATEGORY_MAPPING = {
    "Vagueness": "1",
    "Incompleteness": "2",
    "Lexical ambiguity": "3",
    "Relational tacit knowledge": "1",
    "Collective tacit knowledge": "2",
    "One-to-many mapping to design": "1",
    "No direct mapping to design": "2",
}


# ---------- Schema utilities ----------

def add_ground_truth_answer_column(cursor):
    try:
        cursor.execute(
            "ALTER TABLE generated_questions ADD COLUMN ground_truth_answer TEXT"
        )
    except sqlite3.OperationalError:
        pass


# ---------- Update logic ----------

def update_boolean_answers(cursor):
    """
    Boolean correct answers:
    - atomic-rule questions → rules.rule_id
    - parent-rule questions → rules.parent_rule_id (inherit from atomic rules)
    """

    # --- 1. Atomic rule questions ---
    cursor.execute("""
        UPDATE generated_questions
        SET ground_truth_answer = (
            SELECT ambiguity
            FROM rules r
            WHERE r.rule_id = generated_questions.rule_id)
        WHERE template_id IN (
            SELECT template_id
            FROM templates
            WHERE answer_type = 'bool'
              AND question_template LIKE '%rule_text_atomic%'
        );
    """)

    # --- 2. Parent rule questions ---
    cursor.execute("""
        UPDATE generated_questions
        SET ground_truth_answer = (
            SELECT r.ambiguity
            FROM rules r
            WHERE r.parent_rule_id = generated_questions.parent_rule_id
            LIMIT 1
        )
        WHERE generated_questions.template_id IN (
            SELECT template_id
            FROM templates
            WHERE answer_type = 'bool'
              AND question_template LIKE '%parent_rule_text%'
        )
    """)


def update_multichoice_main_categories(cursor):
    """
    Multichoice questions about MAIN categories ("categories" in template).
    """
    cursor.execute("""
        UPDATE generated_questions
        SET ground_truth_answer = (
            SELECT
                CASE r.classification
                    WHEN 'Vagueness' THEN '1'
                    WHEN 'Incompleteness' THEN '1'
                    WHEN 'Lexical ambiguity' THEN '1'
                    WHEN 'Relational Tacit Knowledge' THEN '2'
                    WHEN 'Collective Tacit Knowledge' THEN '2'
                    WHEN 'One-to-many mapping' THEN '3'
                    WHEN 'No direct mapping to design' THEN '3'
                    ELSE '4'
                END
            FROM rules r
            WHERE r.rule_id = generated_questions.rule_id
        )
        WHERE template_id IN (
            SELECT template_id
            FROM templates
            WHERE answer_type = 'multichoice'
                AND question_template LIKE '%categor%'
        );
    """)


def update_multichoice_subcategories(cursor):
    """
    Multichoice questions about SUB-categories.
    """

    # First: Prescriptive / Definition → none of them (4)
    cursor.execute("""
        UPDATE generated_questions
        SET ground_truth_answer = '4'
        WHERE rule_id IN (
            SELECT rule_id
            FROM rules
            WHERE classification IN ('Prescriptive', 'Definition')
        )
        AND template_id IN (
            SELECT template_id
            FROM templates
            WHERE answer_type = 'multichoice'
              AND question_template LIKE '%sub-categories%'
        )
    """)

    # Then: valid sub-categories
    for subcat, digit in SUBCATEGORY_MAPPING.items():
        cursor.execute("""
            UPDATE generated_questions
            SET ground_truth_answer = ?
            WHERE rule_id IN (
                SELECT rule_id
                FROM rules
                WHERE classification = ?
            )
            AND template_id IN (
                SELECT template_id
                FROM templates
                WHERE answer_type = 'multichoice'
                  AND question_template LIKE '%sub-categories%'
            )
        """, (digit, subcat))


# ---------- Main ----------

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    add_ground_truth_answer_column(cursor)

    update_boolean_answers(cursor)
    update_multichoice_main_categories(cursor)
    update_multichoice_subcategories(cursor)

    conn.commit()
    conn.close()

    print("✔ ground_truth_answer successfully generated")


if __name__ == "__main__":
    main()
