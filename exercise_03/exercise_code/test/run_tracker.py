
import time
import torch
def run_tracker(sequences, db, tracker, device, verbose=False):
    # run the tracker on the database with preprocessed frames
    time_total = 0
    results_seq = {}
    for seq_idx, seq in enumerate(sequences):
        tracker.reset()
        now = time.time()

        if verbose:
            print(f"Tracking {seq_idx}: {seq}")

        # Get the resulting tracks for the sequence
        with torch.no_grad():
            for frame in db[str(seq)]:
                tracker.step(frame)
        results = tracker.get_results()
        results_seq[str(seq)] = results

        time_total += time.time() - now

        if verbose:
            print(f"Tracks found: {len(results)}")
            print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

    print(f"Runtime for all sequences: {time_total:.1f} s.")
    return results_seq
