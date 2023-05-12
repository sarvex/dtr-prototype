import ray
from tqdm import tqdm


def get_futures(futures, desc="Jobs", progress_bar=True):
    if not progress_bar:
        return ray.get(futures)
    results = []
    with tqdm(total=len(futures), desc=desc) as pbar:
        while len(futures):
            done_results, futures = ray.wait(futures)
            results.extend(ray.get(done_results))
            pbar.update((len(done_results)))
    return results
