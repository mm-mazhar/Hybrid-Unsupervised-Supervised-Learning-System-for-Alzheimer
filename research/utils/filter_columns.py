from typing import List


def validate_and_filter_columns(
    columns_to_validate: List[str], all_available_columns: List[str]
) -> List[str]:
    """
    Validates a list of column names against a main list of available columns.

    It returns a new list containing only the column names from the first list
    that are present in the second (main) list. It also prints a message
    identifying any columns that were removed.

    Args:
        columns_to_validate (List[str]): The small list of columns to check and update.
        all_available_columns (List[str]): The large, main list of all valid column names.

    Returns:
        List[str]: A new, filtered list containing only the valid column names.
    """
    # For efficiency, convert the large list of available columns to a set.
    # Checking for an item's existence in a set is much faster than in a list.
    available_columns_set = set(all_available_columns)

    # Use list comprehensions to efficiently find valid and invalid columns
    valid_columns = [col for col in columns_to_validate if col in available_columns_set]
    invalid_columns = [
        col for col in columns_to_validate if col not in available_columns_set
    ]

    # Provide feedback to the user about what was changed
    if invalid_columns:
        print("--- Column List Update ---")
        print(
            "The following columns were NOT found in the main list and have been removed:"
        )
        for col in invalid_columns:
            print(f"- {col}")
        print("--------------------------")
    else:
        print("All columns were found in the main list. No updates were needed.")

    return valid_columns
