import pandas as pd
from pathlib import Path

BATCHE_DIR = Path(__file__).parent.parent / "data" / "batches"

def load_batch(batch_number: int) -> pd.DataFrame:
    """
    Load cumulative training data up to and including batch_number.

    Args:
        batch_number: integer 1, 2, or 3

    Returns:
        DataFrame with all rows from batch 1 to batch_number combined.
    """
    if batch_number not in (1, 2, 3):
        raise ValueError(f"batch_number must be 1, 2 or 3. Got {batch_number}.")
    
    dfs = []
    for i in range(1, batch_number+1):
        path = BATCHE_DIR / f"batch_{i}.csv"
        dfs.append(pd.read_csv(path))
    
    return pd.concat(dfs, ignore_index=True)
        